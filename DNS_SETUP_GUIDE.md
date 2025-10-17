# DNS Setup Guide - red-team-tools.thegavl.com
**Quick Setup: 5-10 minutes**

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

---

## 🌐 Step 1: Log Into Namecheap

1. Go to https://namecheap.com
2. Click **"Sign In"** (top right)
3. Enter your credentials
4. You should see your dashboard

---

## 📋 Step 2: Access DNS Settings

1. Click **"Domain List"** in the left sidebar
2. Find **thegavl.com** in your list
3. Click **"Manage"** button next to thegavl.com
4. Click the **"Advanced DNS"** tab

---

## ⚙️ Step 3: Configure DNS Records

### Option A: Deploying to Bluehost (Recommended)

You already have Bluehost, so this is the easiest option.

**Add this record:**

```
Type:     CNAME
Host:     red-team-tools
Value:    your-bluehost-domain.com
TTL:      Automatic
```

**To find your Bluehost domain:**
1. Log into Bluehost cPanel
2. Look for "Primary Domain" or "Main Domain"
3. It's usually something like: `yourusername.bluehost.com` or your actual domain

**Example:**
```
Type:     CNAME
Host:     red-team-tools
Value:    thegavl.com
TTL:      Automatic
```

### Option B: Deploying to GitHub Pages (Like aios.is)

If you want free hosting:

```
Type:     CNAME
Host:     red-team-tools
Value:    workofarttattoo.github.io
TTL:      Automatic
```

Then create a CNAME file in your repo:
```bash
echo "red-team-tools.thegavl.com" > CNAME
git add CNAME
git commit -m "Add CNAME for red-team-tools subdomain"
git push
```

### Option C: Custom Server/VPS

If you have a dedicated server:

```
Type:     A Record
Host:     red-team-tools
Value:    YOUR.SERVER.IP.ADDRESS
TTL:      Automatic
```

---

## 🖱️ Step 4: Add the Record in Namecheap

1. In the **"Advanced DNS"** tab, scroll to **"Host Records"** section
2. Click **"ADD NEW RECORD"** button
3. Select record type: **CNAME** (or A if using custom server)
4. Enter **Host**: `red-team-tools`
5. Enter **Value**: (see options above)
6. Leave **TTL**: `Automatic`
7. Click the **green checkmark** ✅ to save

---

## ⏱️ Step 5: Wait for DNS Propagation

**Typical wait time:** 15-60 minutes (sometimes up to 24 hours)

### Check if DNS is working:

**Option 1: Use dig command (Mac/Linux)**
```bash
dig red-team-tools.thegavl.com
```

Look for the answer section - it should show your Bluehost domain or IP.

**Option 2: Use online tool**
- Go to https://dnschecker.org
- Enter: `red-team-tools.thegavl.com`
- Click "Search"
- You'll see propagation status worldwide

**Option 3: Try accessing in browser**
```
http://red-team-tools.thegavl.com
```

Initially you might see:
- "Site not found" - DNS not propagated yet (wait)
- "Directory listing" or "Coming soon" - DNS working, need to deploy files
- 404 error - DNS working, files not uploaded

---

## ✅ Step 6: Verify DNS is Working

Once DNS propagates, test:

```bash
# Check DNS resolution
dig red-team-tools.thegavl.com +short

# Check if site responds
curl -I http://red-team-tools.thegavl.com
```

Expected output:
```
HTTP/1.1 200 OK
Server: Apache/nginx
```

---

## 🔒 Step 7: Enable SSL (HTTPS)

### If using Bluehost:

1. Log into Bluehost cPanel
2. Go to **"Security"** section
3. Click **"SSL/TLS Status"**
4. Find **red-team-tools.thegavl.com** in the list
5. Click **"Run AutoSSL"**
6. Wait 5-10 minutes
7. SSL certificate will be automatically installed

### If using GitHub Pages:

GitHub automatically provides SSL via Let's Encrypt:
1. Go to repo settings → Pages
2. Check **"Enforce HTTPS"** box
3. Wait 10-15 minutes for certificate provisioning

### If using custom server:

Use Let's Encrypt:
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-apache

# Generate certificate
sudo certbot --apache -d red-team-tools.thegavl.com

# Auto-renewal is configured automatically
```

---

## 🎯 Final Verification

Once everything is set up:

```bash
# Test HTTP
curl -I http://red-team-tools.thegavl.com

# Test HTTPS
curl -I https://red-team-tools.thegavl.com

# Check SSL certificate
openssl s_client -connect red-team-tools.thegavl.com:443 -servername red-team-tools.thegavl.com
```

Expected:
- ✅ HTTP redirects to HTTPS
- ✅ HTTPS loads without warnings
- ✅ Certificate is valid (not self-signed)

---

## 🔧 Troubleshooting

### Problem: DNS not resolving after 24 hours

**Causes:**
- Typo in DNS record
- Wrong nameservers configured

**Fix:**
1. Check nameservers in Namecheap:
   - Go to Domain List → thegavl.com → Domain tab
   - Should say "Namecheap BasicDNS" or "Namecheap PremiumDNS"
   - If it says something else, click "Change" and select Namecheap BasicDNS
2. Double-check the CNAME record:
   - Host: `red-team-tools` (no @ symbol, no trailing dot)
   - Value: `thegavl.com` or `workofarttattoo.github.io` (no http://, no trailing slash)

### Problem: DNS works but site shows 404

**Causes:**
- Files not uploaded to Bluehost
- Wrong directory structure

**Fix:**
1. Log into Bluehost cPanel
2. Go to File Manager
3. Navigate to `public_html/`
4. Create subdirectory: `red-team-tools/`
5. Upload your files there
6. Make sure there's an `index.html` or `index.php` file

### Problem: SSL certificate not working

**Causes:**
- DNS not fully propagated
- AutoSSL not run yet
- Mixed content (HTTP resources on HTTPS page)

**Fix:**
1. Wait 24 hours after DNS propagation for AutoSSL
2. Manually trigger SSL in Bluehost cPanel → SSL/TLS Status → Run AutoSSL
3. Check browser console for mixed content warnings
4. Ensure all resources (images, CSS, JS) use HTTPS or relative URLs

### Problem: "This site can't provide a secure connection"

**Causes:**
- Accessing via HTTPS before SSL certificate is installed

**Fix:**
1. Access via HTTP first: `http://red-team-tools.thegavl.com`
2. Wait for SSL certificate to be provisioned
3. Then enforce HTTPS

---

## 📧 DNS Record Summary (For Reference)

After setup, your DNS should have:

```
thegavl.com Domain Records:

Type    Host                Value                           TTL
A       @                   YOUR_BLUEHOST_IP               Automatic
CNAME   www                 thegavl.com                    Automatic
CNAME   red-team-tools      thegavl.com                    Automatic
CNAME   aios                workofarttattoo.github.io      Automatic (if you have aios.thegavl.com)
```

---

## 🎉 Success Checklist

- [ ] DNS record added in Namecheap
- [ ] Waited for DNS propagation (15-60 minutes)
- [ ] DNS resolves correctly (`dig red-team-tools.thegavl.com`)
- [ ] Site accessible via HTTP
- [ ] SSL certificate installed
- [ ] Site accessible via HTTPS
- [ ] No browser warnings (padlock icon green)
- [ ] Ready to deploy files!

---

## 📞 Need Help?

**Namecheap Support:**
- Live Chat: https://www.namecheap.com/support/live-chat/
- Phone: 1-888-401-4678
- Available 24/7

**Bluehost Support:**
- Phone: 1-888-401-4678
- Live Chat: Available in cPanel
- Available 24/7

**DNS Checker Tools:**
- https://dnschecker.org
- https://www.whatsmydns.net
- https://mxtoolbox.com/SuperTool.aspx

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
