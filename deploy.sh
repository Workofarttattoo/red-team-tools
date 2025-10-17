#!/bin/bash
# Red Team Tools Deployment Script
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "🚀 Deploying Red Team Tools Site..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "index.html" ]; then
    echo -e "${RED}Error: index.html not found. Run from red-team-tools directory${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Checking git status...${NC}"

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Red Team Security Tools site"
else
    echo "Git repository already initialized"
    git add .
    git commit -m "Update: Red Team Tools site $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"
fi

echo -e "${YELLOW}Step 2: Setting up GitHub repository...${NC}"

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo -e "${RED}GitHub CLI (gh) not found. Installing...${NC}"
    brew install gh || {
        echo -e "${RED}Failed to install gh. Please install manually: brew install gh${NC}"
        exit 1
    }
fi

# Check if logged in to GitHub
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}Please log in to GitHub:${NC}"
    gh auth login
fi

# Create or update GitHub repo
REPO_NAME="red-team-tools"
echo "Creating/updating GitHub repository: $REPO_NAME"

# Check if repo exists
if gh repo view "$REPO_NAME" &> /dev/null; then
    echo "Repository already exists, pushing updates..."
    git branch -M main
    git push -u origin main
else
    echo "Creating new repository..."
    gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
fi

echo -e "${YELLOW}Step 3: Enabling GitHub Pages...${NC}"

# Enable GitHub Pages
gh api repos/:owner/$REPO_NAME/pages \
    -X POST \
    -f source[branch]=main \
    -f source[path]=/ \
    2>/dev/null || echo "GitHub Pages may already be enabled"

echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo ""
echo -e "${GREEN}Your site will be available at:${NC}"
echo -e "${GREEN}https://$(gh api user --jq .login).github.io/$REPO_NAME${NC}"
echo ""
echo -e "${YELLOW}Note: It may take a few minutes for GitHub Pages to build and deploy.${NC}"
echo ""
echo -e "${YELLOW}To set up a custom domain:${NC}"
echo "1. Go to: https://github.com/$(gh api user --jq .login)/$REPO_NAME/settings/pages"
echo "2. Add your custom domain (e.g., red-team-tools.aios.is)"
echo "3. Update your DNS records to point to GitHub Pages"
echo ""
