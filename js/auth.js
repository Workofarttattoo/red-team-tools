/**
 * Authentication Module
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. Patents filed..
 */

class AuthManager {
    constructor() {
        this.supabase = supabaseClient;
        this.session = null;
        this.loginAttempts = {};
        this.initializeAuth();
    }

    async initializeAuth() {
        if (!this.supabase) {
            console.error('Supabase client not initialized');
            return;
        }

        // Get initial session
        const { data: { session } } = await this.supabase.auth.getSession();
        this.session = session;

        // Listen for auth changes
        this.supabase.auth.onAuthStateChange((event, session) => {
            this.session = session;
            this.handleAuthChange(event, session);
        });
    }

    handleAuthChange(event, session) {
        switch (event) {
            case 'SIGNED_IN':
                this.logAuditEvent('login', session.user);
                break;
            case 'SIGNED_OUT':
                this.logAuditEvent('logout');
                break;
            case 'TOKEN_REFRESHED':
                this.logAuditEvent('token_refresh', session.user);
                break;
            case 'USER_UPDATED':
                this.logAuditEvent('user_update', session.user);
                break;
        }
    }

    // Login with email and password
    async login(email, password, rememberMe = false) {
        try {
            // Check for rate limiting
            if (!this.checkRateLimit(email, 'login')) {
                throw new Error('Too many login attempts. Please try again later.');
            }

            // Validate input
            if (!this.validateEmail(email)) {
                throw new Error('Please enter a valid email address.');
            }

            if (!password || password.length < AUTH_CONFIG.PASSWORD_MIN_LENGTH) {
                throw new Error(`Password must be at least ${AUTH_CONFIG.PASSWORD_MIN_LENGTH} characters.`);
            }

            // Attempt login
            const { data, error } = await this.supabase.auth.signInWithPassword({
                email: email.toLowerCase().trim(),
                password: password
            });

            if (error) {
                this.recordFailedAttempt(email);
                throw error;
            }

            // Clear failed attempts on successful login
            this.clearFailedAttempts(email);

            // Store remember me preference
            if (rememberMe) {
                localStorage.setItem('rememberMe', 'true');
            }

            // Log successful login
            this.logAuditEvent('login_success', data.user);

            return { success: true, user: data.user, session: data.session };

        } catch (error) {
            console.error('Login error:', error);
            this.logAuditEvent('login_failed', null, { email, error: error.message });
            throw error;
        }
    }

    // Sign up new user
    async signup(email, password, metadata = {}) {
        try {
            // Check for rate limiting
            if (!this.checkRateLimit(email, 'signup')) {
                throw new Error('Too many signup attempts. Please try again later.');
            }

            // Validate input
            if (!this.validateEmail(email)) {
                throw new Error('Please enter a valid email address.');
            }

            if (!this.validatePassword(password)) {
                throw new Error(this.getPasswordRequirements());
            }

            // Check if signup is allowed
            if (!AUTH_CONFIG.FEATURES.allowSignup) {
                throw new Error('New registrations are currently disabled.');
            }

            // Attempt signup
            const { data, error } = await this.supabase.auth.signUp({
                email: email.toLowerCase().trim(),
                password: password,
                options: {
                    data: {
                        ...metadata,
                        signup_timestamp: new Date().toISOString(),
                        signup_ip: await this.getClientIP()
                    },
                    emailRedirectTo: `${AUTH_CONFIG.APP_DOMAIN}/verify-email.html`
                }
            });

            if (error) throw error;

            // Log signup
            this.logAuditEvent('signup_success', data.user);

            return { success: true, user: data.user, requiresVerification: AUTH_CONFIG.REQUIRE_EMAIL_VERIFICATION };

        } catch (error) {
            console.error('Signup error:', error);
            this.logAuditEvent('signup_failed', null, { email, error: error.message });
            throw error;
        }
    }

    // Logout
    async logout() {
        try {
            const { error } = await this.supabase.auth.signOut();
            if (error) throw error;

            // Clear local storage
            localStorage.removeItem('rememberMe');
            sessionStorage.clear();

            this.logAuditEvent('logout_success');
            return { success: true };

        } catch (error) {
            console.error('Logout error:', error);
            this.logAuditEvent('logout_failed', null, { error: error.message });
            throw error;
        }
    }

    // Password reset
    async resetPassword(email) {
        try {
            // Check for rate limiting
            if (!this.checkRateLimit(email, 'passwordReset')) {
                throw new Error('Too many password reset attempts. Please try again later.');
            }

            if (!this.validateEmail(email)) {
                throw new Error('Please enter a valid email address.');
            }

            const { error } = await this.supabase.auth.resetPasswordForEmail(
                email.toLowerCase().trim(),
                {
                    redirectTo: `${AUTH_CONFIG.APP_DOMAIN}/reset-password.html`
                }
            );

            if (error) throw error;

            this.logAuditEvent('password_reset_requested', null, { email });
            return { success: true };

        } catch (error) {
            console.error('Password reset error:', error);
            this.logAuditEvent('password_reset_failed', null, { email, error: error.message });
            throw error;
        }
    }

    // Update password
    async updatePassword(newPassword) {
        try {
            if (!this.validatePassword(newPassword)) {
                throw new Error(this.getPasswordRequirements());
            }

            const { data, error } = await this.supabase.auth.updateUser({
                password: newPassword
            });

            if (error) throw error;

            this.logAuditEvent('password_updated', data.user);
            return { success: true };

        } catch (error) {
            console.error('Password update error:', error);
            throw error;
        }
    }

    // OAuth login
    async loginWithOAuth(provider) {
        try {
            if (!AUTH_CONFIG.FEATURES.allowOAuth) {
                throw new Error('OAuth login is currently disabled.');
            }

            if (!AUTH_CONFIG.OAUTH_PROVIDERS[provider]) {
                throw new Error(`${provider} login is not configured.`);
            }

            const { data, error } = await this.supabase.auth.signInWithOAuth({
                provider: provider,
                options: {
                    redirectTo: `${AUTH_CONFIG.APP_DOMAIN}/dashboard.html`
                }
            });

            if (error) throw error;

            this.logAuditEvent('oauth_login_initiated', null, { provider });
            return { success: true, url: data.url };

        } catch (error) {
            console.error('OAuth login error:', error);
            throw error;
        }
    }

    // Get current session
    async getSession() {
        const { data: { session } } = await this.supabase.auth.getSession();
        return session;
    }

    // Get current user
    async getCurrentUser() {
        const { data: { user } } = await this.supabase.auth.getUser();
        return user;
    }

    // Check if user is authenticated
    async isAuthenticated() {
        const session = await this.getSession();
        return !!session;
    }

    // Validate email format
    validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    // Validate password strength
    validatePassword(password) {
        if (!password || password.length < AUTH_CONFIG.PASSWORD_MIN_LENGTH) {
            return false;
        }

        // Check for at least one uppercase, one lowercase, one number, and one special character
        const hasUpperCase = /[A-Z]/.test(password);
        const hasLowerCase = /[a-z]/.test(password);
        const hasNumbers = /\d/.test(password);
        const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

        return hasUpperCase && hasLowerCase && hasNumbers && hasSpecialChar;
    }

    // Get password requirements message
    getPasswordRequirements() {
        return `Password must be at least ${AUTH_CONFIG.PASSWORD_MIN_LENGTH} characters and include:
        - One uppercase letter
        - One lowercase letter
        - One number
        - One special character (!@#$%^&*(),.?":{}|<>)`;
    }

    // Rate limiting
    checkRateLimit(identifier, action) {
        const key = `rate_${action}_${identifier}`;
        const now = Date.now();
        const attempts = JSON.parse(localStorage.getItem(key) || '[]');

        // Clean old attempts (older than 1 minute)
        const recentAttempts = attempts.filter(time => now - time < 60000);

        const limit = AUTH_CONFIG.RATE_LIMIT[action] || 10;
        if (recentAttempts.length >= limit) {
            return false;
        }

        recentAttempts.push(now);
        localStorage.setItem(key, JSON.stringify(recentAttempts));
        return true;
    }

    // Track failed login attempts
    recordFailedAttempt(email) {
        const key = `failed_attempts_${email}`;
        const attempts = JSON.parse(localStorage.getItem(key) || '{"count": 0, "lastAttempt": 0}');

        attempts.count++;
        attempts.lastAttempt = Date.now();

        localStorage.setItem(key, JSON.stringify(attempts));

        // Check if account should be locked
        if (attempts.count >= AUTH_CONFIG.MAX_LOGIN_ATTEMPTS) {
            this.lockAccount(email);
        }
    }

    // Clear failed attempts
    clearFailedAttempts(email) {
        localStorage.removeItem(`failed_attempts_${email}`);
        localStorage.removeItem(`account_locked_${email}`);
    }

    // Lock account temporarily
    lockAccount(email) {
        const until = Date.now() + AUTH_CONFIG.LOCKOUT_DURATION;
        localStorage.setItem(`account_locked_${email}`, until.toString());
    }

    // Check if account is locked
    isAccountLocked(email) {
        const lockedUntil = localStorage.getItem(`account_locked_${email}`);
        if (!lockedUntil) return false;

        const until = parseInt(lockedUntil);
        if (Date.now() < until) {
            return true;
        }

        // Unlock if time has passed
        localStorage.removeItem(`account_locked_${email}`);
        return false;
    }

    // Get client IP (for audit logging)
    async getClientIP() {
        try {
            const response = await fetch('https://api.ipify.org?format=json');
            const data = await response.json();
            return data.ip;
        } catch {
            return 'unknown';
        }
    }

    // Audit logging
    logAuditEvent(event, user = null, metadata = {}) {
        if (!AUTH_CONFIG.FEATURES.enableAuditLog) return;

        const auditLog = JSON.parse(localStorage.getItem('audit_log') || '[]');
        const entry = {
            event,
            timestamp: new Date().toISOString(),
            userId: user?.id || null,
            userEmail: user?.email || null,
            metadata
        };

        auditLog.push(entry);

        // Keep only last 1000 entries
        if (auditLog.length > 1000) {
            auditLog.shift();
        }

        localStorage.setItem('audit_log', JSON.stringify(auditLog));

        // In production, also send to backend
        if (this.supabase && user) {
            this.supabase.from('audit_logs').insert([entry]).catch(() => {});
        }
    }
}

// Export for use in other files
if (typeof window !== 'undefined') {
    window.AuthManager = AuthManager;
}