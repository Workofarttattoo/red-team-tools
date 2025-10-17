# Deployment Guide for thegavl.com

## ✅ File Permissions Fixed

The `js` directory permissions have been corrected from `700` to `755`, allowing web server access.

## 🚀 Deployment Steps

### 1. Files to Deploy

Upload ALL files from `/Users/noone/aios/red-team-tools/web-deploy/` to your web server:

```
Required Files:
├── index.html                    # Landing page
├── login.html                    # Login page
├── register.html                 # Registration page
├── dashboard.html                # Main dashboard
├── js/                          # JavaScript files (REQUIRED)
│   ├── auth-config.js           # Supabase configuration
│   └── auth.js                  # Authentication logic
├── burp-suite.html              # Tool pages
├── console-monitor.html
├── directory-fuzzer.html
├── hash-cracker.html
├── reverse-shell.html
├── shodan-search.html
├── sqlmap.html
├── tech-stack-analyzer.html
└── test-auth.html              # For testing authentication
```

### 2. Web Server Configuration

#### For Apache (.htaccess)
Create `.htaccess` file in root:
```apache
# Enable HTTPS redirect
RewriteEngine On
RewriteCond %{HTTPS} off
RewriteRule ^(.*)$ https://%{HTTP_HOST}/$1 [R=301,L]

# Security headers
Header set X-Content-Type-Options "nosniff"
Header set X-Frame-Options "DENY"
Header set X-XSS-Protection "1; mode=block"
Header set Referrer-Policy "strict-origin-when-cross-origin"

# Cache static assets
<FilesMatch "\.(css|js|jpg|jpeg|png|gif|ico)$">
    Header set Cache-Control "max-age=604800, public"
</FilesMatch>
```

#### For Nginx
```nginx
server {
    listen 443 ssl http2;
    server_name thegavl.com www.thegavl.com;

    root /path/to/web-deploy;
    index index.html;

    # SSL configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Security headers
    add_header X-Content-Type-Options "nosniff";
    add_header X-Frame-Options "DENY";
    add_header X-XSS-Protection "1; mode=block";

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(css|js|jpg|jpeg|png|gif|ico)$ {
        expires 7d;
        add_header Cache-Control "public, immutable";
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name thegavl.com www.thegavl.com;
    return 301 https://$server_name$request_uri;
}
```

### 3. CORS Configuration for Supabase

In Supabase Dashboard:
1. Go to Settings → API
2. Add your domain to CORS allowed origins:
   - `https://thegavl.com`
   - `https://www.thegavl.com`

### 4. Update Authentication Config

If needed, update `js/auth-config.js` redirect URLs to use full domain:
```javascript
REDIRECT_URLS: {
    afterLogin: 'https://thegavl.com/dashboard.html',
    afterLogout: 'https://thegavl.com/index.html',
    afterSignup: 'https://thegavl.com/verify-email.html',
    passwordReset: 'https://thegavl.com/reset-password.html'
}
```

### 5. SSL/HTTPS Setup

**REQUIRED** - Supabase authentication requires HTTPS

1. **Using Let's Encrypt (Free)**:
   ```bash
   sudo certbot --apache -d thegavl.com -d www.thegavl.com
   # or for nginx:
   sudo certbot --nginx -d thegavl.com -d www.thegavl.com
   ```

2. **Using Cloudflare (Free)**:
   - Add site to Cloudflare
   - Enable "Full SSL/TLS" mode
   - Enable "Always Use HTTPS"

### 6. Test Deployment

After deployment, test:

1. **Basic Access**: https://thegavl.com should load
2. **JavaScript Loading**: Check browser console for errors
3. **Authentication Test**: https://thegavl.com/test-auth.html
4. **Login Flow**: Try creating an account and logging in

### 7. Common Issues & Solutions

#### Issue: JavaScript files not loading (404)
- **Solution**: Ensure `js/` directory was uploaded with correct permissions (755)

#### Issue: CORS errors in console
- **Solution**: Add your domain to Supabase CORS settings

#### Issue: Authentication redirects not working
- **Solution**: Update redirect URLs in `js/auth-config.js` to use full domain

#### Issue: "Mixed content" errors
- **Solution**: Ensure ALL resources are loaded over HTTPS

#### Issue: Supabase connection fails
- **Solution**:
  - Check that HTTPS is working
  - Verify Supabase project is active (not paused)
  - Check browser console for specific errors

### 8. Security Checklist

- [ ] HTTPS enabled and forced
- [ ] File permissions correct (755 for directories, 644 for files)
- [ ] No `.env` or sensitive files uploaded
- [ ] Security headers configured
- [ ] CORS properly configured in Supabase
- [ ] Rate limiting enabled (via Cloudflare or server config)

### 9. Monitoring

Set up monitoring to ensure uptime:
- **UptimeRobot** (free): https://uptimerobot.com
- **Cloudflare Analytics** (if using Cloudflare)
- **Google Analytics** (optional)

## 📱 Mobile Responsiveness

The site is mobile-responsive, but test on actual devices:
- iPhone Safari
- Android Chrome
- Tablet browsers

## 🎯 Quick Deployment Commands

### Via FTP/SFTP
```bash
# Using rsync (recommended)
rsync -avz --exclude='.env' --exclude='*.md' /Users/noone/aios/red-team-tools/web-deploy/ user@thegavl.com:/var/www/html/

# Using scp
scp -r /Users/noone/aios/red-team-tools/web-deploy/* user@thegavl.com:/var/www/html/
```

### Via Git (if server has git)
```bash
# On server
cd /var/www/html
git clone https://github.com/Workofarttattoo/AioS.git
cp -r AioS/red-team-tools/web-deploy/* .
rm -rf AioS
```

## 🔍 Verification

Once deployed, verify everything works:
1. Visit https://thegavl.com
2. Check browser console (F12) for any errors
3. Test login/signup flow
4. Test each tool page loads correctly
5. Verify Supabase connection on test page

---

If you encounter issues, check:
- Browser Developer Console (F12)
- Server error logs
- Supabase Dashboard logs
