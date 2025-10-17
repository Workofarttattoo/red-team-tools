# Production Authentication Setup Guide

## Quick Start

To enable production authentication for the Red Team Tools suite, follow these steps:

### 1. Create a Supabase Account

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up for a free account
3. Create a new project
4. Note your project URL and anon key from Settings → API

### 2. Configure Authentication

Edit `js/auth-config.js` and update these values:

```javascript
const AUTH_CONFIG = {
    // Replace with your Supabase project details
    SUPABASE_URL: 'https://your-project-id.supabase.co',
    SUPABASE_ANON_KEY: 'your-anon-public-key-here',
    // ... other settings
};
```

### 3. Set Up Database Tables (Optional but Recommended)

Run this SQL in Supabase SQL Editor to create audit logging:

```sql
-- Create audit logs table
CREATE TABLE audit_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event TEXT NOT NULL,
    user_id UUID REFERENCES auth.users(id),
    user_email TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Create policy for users to insert their own logs
CREATE POLICY "Users can insert their own audit logs"
    ON audit_logs FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

-- Create index for performance
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_event ON audit_logs(event);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
```

### 4. Configure Email Templates

In Supabase Dashboard:

1. Go to Authentication → Email Templates
2. Customize the following templates:
   - Confirm Email
   - Reset Password
   - Magic Link

### 5. Set Up Authentication Providers (Optional)

To enable OAuth login:

1. Go to Authentication → Providers
2. Enable desired providers:
   - Google
   - GitHub
   - GitLab
   - Bitbucket
3. Add OAuth credentials for each provider
4. Update `js/auth-config.js` to enable OAuth:

```javascript
OAUTH_PROVIDERS: {
    google: true,    // Set to true after configuring in Supabase
    github: true,
    gitlab: false,
    bitbucket: false
}
```

### 6. Configure Security Settings

In Supabase Dashboard:

1. **Enable Email Verification**:
   - Authentication → Settings → Enable email confirmations

2. **Set Password Requirements**:
   - Authentication → Settings → Password minimum length

3. **Configure Rate Limiting**:
   - Authentication → Settings → Rate limits

4. **Set Up CORS**:
   - Settings → API → CORS Configuration
   - Add your domain: `https://yourdomain.com`

### 7. Environment Variables (For CI/CD)

For production deployment, use environment variables instead of hardcoding:

```bash
# .env file (DO NOT COMMIT)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
```

Then in your deployment script:

```javascript
// js/auth-config.js
const AUTH_CONFIG = {
    SUPABASE_URL: process.env.SUPABASE_URL || 'https://your-project.supabase.co',
    SUPABASE_ANON_KEY: process.env.SUPABASE_ANON_KEY || 'your-anon-key-here',
    // ...
};
```

## Production Checklist

- [ ] Supabase project created
- [ ] Authentication configured in `js/auth-config.js`
- [ ] Email templates customized
- [ ] Email verification enabled
- [ ] Password policy configured
- [ ] Rate limiting enabled
- [ ] CORS configured for production domain
- [ ] RLS (Row Level Security) enabled on all tables
- [ ] Audit logging table created
- [ ] OAuth providers configured (if needed)
- [ ] Environment variables set up for deployment
- [ ] SSL certificate configured for production domain
- [ ] Regular backup policy in place

## Security Best Practices

1. **Never commit credentials**: Use environment variables
2. **Enable RLS**: Always enable Row Level Security on tables
3. **Use HTTPS**: Always serve over HTTPS in production
4. **Rate Limiting**: Configure aggressive rate limits
5. **Email Verification**: Always require email verification
6. **Strong Passwords**: Enforce minimum 8 characters with complexity
7. **Session Management**: Use short session durations for sensitive apps
8. **Audit Logging**: Log all authentication events
9. **Regular Updates**: Keep Supabase SDK updated
10. **Monitor Usage**: Set up alerts for unusual activity

## Testing

After configuration:

1. Test user registration
2. Test email verification
3. Test password reset
4. Test login/logout
5. Test session persistence
6. Test rate limiting
7. Test OAuth providers (if configured)

## Troubleshooting

### "Authentication not configured" Error

- Verify `SUPABASE_URL` and `SUPABASE_ANON_KEY` are correct
- Check browser console for specific error messages
- Ensure Supabase project is active (not paused)

### Email Not Sending

- Check email templates in Supabase
- Verify email provider settings
- Check spam folder
- Enable email logs in Supabase

### OAuth Not Working

- Verify OAuth app credentials
- Check redirect URLs match exactly
- Ensure provider is enabled in Supabase

### Session Issues

- Clear browser localStorage and cookies
- Check session duration settings
- Verify JWT secret hasn't changed

## Support

For issues specific to:
- **Supabase**: https://supabase.com/docs
- **Red Team Tools**: Contact support@thegavl.com

---

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.