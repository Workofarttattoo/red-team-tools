/**
 * Authentication Configuration
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 *
 * SETUP INSTRUCTIONS:
 * 1. Create a Supabase account at https://supabase.com
 * 2. Create a new project
 * 3. Go to Settings -> API
 * 4. Copy your project URL and anon/public key
 * 5. Replace the values below
 *
 * For production deployment:
 * - Use environment variables instead of hardcoding
 * - Enable RLS (Row Level Security) in Supabase
 * - Configure email templates in Authentication settings
 * - Set up proper CORS and domain restrictions
 */

const AUTH_CONFIG = {
    // Supabase Configuration
    SUPABASE_URL: process.env.SUPABASE_URL || 'https://trokobwiphidmrmhwkni.supabase.co',
    SUPABASE_ANON_KEY: process.env.SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRyb2tvYndpcGhpZG1ybWh3a25pIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2NTk4MTQsImV4cCI6MjA3NjIzNTgxNH0.D1iTVxtL481Tk6Jr7qSInjOOCZhWmuHT8g-cE_ZT-dM',

    // Application Settings
    APP_NAME: 'TheGAVL Red Team Tools',
    APP_DOMAIN: window.location.origin,

    // Session Settings
    SESSION_DURATION: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
    REMEMBER_ME_DURATION: 30 * 24 * 60 * 60 * 1000, // 30 days

    // Security Settings
    PASSWORD_MIN_LENGTH: 8,
    REQUIRE_EMAIL_VERIFICATION: true,
    ENABLE_MFA: true, // Set to true for production
    MAX_LOGIN_ATTEMPTS: 5,
    LOCKOUT_DURATION: 15 * 60 * 1000, // 15 minutes

    // OAuth Providers (configure in Supabase dashboard)
    OAUTH_PROVIDERS: {
        google: true,
        github: false,
        gitlab: false,
        bitbucket: false
    },

    // Redirect URLs
    REDIRECT_URLS: {
        afterLogin: '/dashboard.html',
        afterLogout: '/index.html',
        afterSignup: '/verify-email.html',
        passwordReset: '/reset-password.html'
    },

    // API Rate Limiting
    RATE_LIMIT: {
        login: 10, // attempts per minute
        signup: 5,
        passwordReset: 3
    },

    // Feature Flags
    FEATURES: {
        allowSignup: true,
        allowPasswordReset: true,
        allowOAuth: false,
        requireTermsAcceptance: true,
        enableAuditLog: true,
        enableSessionManagement: true
    },

    // Compliance & Legal
    COMPLIANCE: {
        gdprEnabled: true,
        ccpaEnabled: true,
        dataRetentionDays: 90,
        requireAge13Plus: true
    }
};

// Validate configuration on load
function validateAuthConfig() {
    const errors = [];

    if (AUTH_CONFIG.SUPABASE_URL === 'https://your-project.supabase.co') {
        errors.push('Supabase URL not configured. Please update AUTH_CONFIG.SUPABASE_URL');
    }

    if (AUTH_CONFIG.SUPABASE_ANON_KEY === 'your-anon-key-here') {
        errors.push('Supabase anon key not configured. Please update AUTH_CONFIG.SUPABASE_ANON_KEY');
    }

    if (errors.length > 0) {
        console.error('⚠️ Authentication Configuration Errors:');
        errors.forEach(error => console.error(`  - ${error}`));
        return false;
    }

    return true;
}

// Initialize Supabase client
let supabaseClient = null;

if (typeof window !== 'undefined' && window.supabase) {
    const { createClient } = window.supabase;
    supabaseClient = createClient(AUTH_CONFIG.SUPABASE_URL, AUTH_CONFIG.SUPABASE_ANON_KEY, {
        auth: {
            autoRefreshToken: true,
            persistSession: true,
            detectSessionInUrl: true,
            flowType: 'pkce' // More secure for SPAs
        },
        global: {
            headers: {
                'X-App-Name': AUTH_CONFIG.APP_NAME
            }
        }
    });
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AUTH_CONFIG, supabaseClient, validateAuthConfig };
}