# Acceptable Use Policy (AUP) - TheGAVL Red Team Security Tools
**Effective Date:** January 1, 2025
**Last Updated:** October 16, 2025

## Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

---

## 1. PURPOSE

This Acceptable Use Policy ("AUP") defines prohibited and acceptable uses of the security assessment tools and penetration testing utilities ("Tools") provided by Corporation of Light DBA TheGAVL ("TheGAVL", "we", "us") at red-team-tools.thegavl.com.

**This AUP is binding and enforceable. Violations may result in immediate account termination, reporting to law enforcement, and legal action.**

---

## 2. ACCEPTABLE USES

### 2.1 Authorized Security Testing

✅ **YOU MAY** use the Tools for:
- **Authorized penetration testing** of systems you own or have written permission to test
- **Vulnerability assessments** with explicit authorization from the system owner
- **Red team exercises** authorized by your organization or client
- **Security research** in controlled, isolated lab environments
- **Educational purposes** on systems specifically designed for training (e.g., HackTheBox, DVWA)
- **Bug bounty programs** where you comply with program rules
- **Compliance audits** (PCI-DSS, HIPAA, SOC 2) with proper authorization
- **Incident response** and forensic investigations with legal authority

### 2.2 Required Authorization

Before using the Tools on any target, you **MUST**:
- ✅ Obtain **explicit written permission** from the system owner or authorized representative
- ✅ Define the **scope** of testing (IP ranges, domains, applications, time windows)
- ✅ Establish **rules of engagement** (allowed/prohibited techniques, emergency procedures)
- ✅ Notify relevant stakeholders (IT teams, security teams, management)
- ✅ Obtain appropriate **liability insurance** if required by contract
- ✅ Maintain documentation of all authorizations

### 2.3 Responsible Disclosure

✅ **YOU MUST**:
- Report discovered vulnerabilities to the affected party promptly
- Allow reasonable time (typically 90 days) for remediation before public disclosure
- Follow the target's responsible disclosure policy if one exists
- Coordinate disclosure timing with the affected party
- Not exploit vulnerabilities for personal gain or malicious purposes

---

## 3. PROHIBITED ACTIVITIES

### 3.1 Unauthorized Access

❌ **YOU MUST NOT** use the Tools for:
- Accessing systems, networks, or data **without explicit authorization**
- Testing systems you do not own without written permission from the owner
- Exceeding the scope of authorized testing
- Continuing testing after authorization has been revoked
- "Drive-by" scanning of random internet hosts
- Testing government systems without proper authorization (illegal under CFAA)

### 3.2 Malicious Activities

❌ **ABSOLUTELY PROHIBITED:**
- Deploying malware, ransomware, or destructive payloads
- Data exfiltration or theft of confidential information
- Denial-of-service (DoS) or distributed denial-of-service (DDoS) attacks
- Defacement or vandalism of websites or systems
- Destruction, alteration, or corruption of data
- Lateral movement beyond authorized scope
- Privilege escalation for unauthorized purposes
- Cryptojacking or cryptocurrency mining on target systems

### 3.3 Criminal Activities

❌ **STRICTLY FORBIDDEN:**
- Hacking for financial gain, extortion, or blackmail
- Identity theft or credential harvesting
- Corporate espionage or stealing trade secrets
- Stalking, harassment, or intimidation using the Tools
- Facilitating any illegal activity
- Violating the Computer Fraud and Abuse Act (CFAA) or equivalent laws
- Testing critical infrastructure without proper authorization
- Interfering with law enforcement or emergency services

### 3.4 Prohibited Targets

❌ **DO NOT SCAN OR TEST:**
- Government systems (federal, state, local) without explicit authorization
- Critical infrastructure (power grids, water systems, healthcare, financial services)
- Educational institutions without authorization
- Third-party systems or services not owned by you or your client
- Systems located in sanctioned countries or jurisdictions
- Military or defense systems
- Nuclear facilities, air traffic control, or emergency services

### 3.5 Abuse of Tools

❌ **YOU MUST NOT:**
- Share your account credentials with others
- Attempt to reverse engineer, decompile, or modify the Tools
- Bypass rate limits, quotas, or security controls
- Automate scanning beyond reasonable limits
- Use the Tools to train AI models without permission
- Resell or sublicense access to the Tools
- Remove copyright notices or attribution

---

## 4. ETHICAL GUIDELINES

### 4.1 Professional Conduct

Security researchers using our Tools must:
- **Act with integrity**: Follow ethical hacking principles
- **Minimize disruption**: Avoid causing service outages or data loss
- **Respect privacy**: Do not access, copy, or disclose personal data unnecessarily
- **Be transparent**: Clearly identify yourself and your intentions when required
- **Follow disclosure norms**: Adhere to industry-standard responsible disclosure timelines

### 4.2 Scope Limitations

**STAY WITHIN SCOPE:**
- Only test systems explicitly authorized in your engagement agreement
- Do not test ancillary systems discovered during testing without additional authorization
- Stop testing immediately if you encounter out-of-scope systems
- Document any out-of-scope discoveries and report them to the client

### 4.3 Dos and Don'ts During Testing

**DO:**
- ✅ Maintain detailed logs of all testing activities
- ✅ Use non-destructive testing methods whenever possible
- ✅ Have an emergency stop procedure in case of issues
- ✅ Coordinate with the client's IT team for production testing
- ✅ Test during agreed-upon time windows
- ✅ Validate findings to eliminate false positives

**DON'T:**
- ❌ Test production systems during business hours without approval
- ❌ Use exploits that could cause system instability or data loss
- ❌ Access sensitive data (PII, PHI, financial records) unless absolutely necessary
- ❌ Share discovered vulnerabilities with third parties before remediation
- ❌ Exaggerate findings to increase engagement value

---

## 5. COMPLIANCE REQUIREMENTS

### 5.1 Legal Compliance

Users must comply with all applicable laws, including:
- **Computer Fraud and Abuse Act (18 U.S.C. § 1030)** - Unauthorized access is a federal crime
- **Stored Communications Act (18 U.S.C. § 2701)** - Accessing stored communications
- **Wiretap Act (18 U.S.C. § 2511)** - Intercepting electronic communications
- **State cybercrime laws** - Many states have additional computer crime statutes
- **Foreign laws** - If testing systems in other jurisdictions
- **GDPR** - If testing systems processing EU resident data
- **HIPAA** - If testing healthcare systems
- **PCI-DSS** - If testing payment card systems

### 5.2 Contractual Obligations

**Before testing, ensure:**
- You have a signed contract or authorization letter
- Liability and indemnification clauses are clear
- Insurance coverage is adequate (E&O, cyber liability)
- Non-disclosure agreements (NDAs) are in place
- Statement of Work (SOW) clearly defines scope

### 5.3 Export Control Compliance

Users must comply with export control laws:
- Do not use the Tools from sanctioned countries
- Do not provide access to sanctioned entities or individuals
- Comply with EAR, ITAR, and UN sanctions
- Report any requests from restricted parties to legal@thegavl.com

---

## 6. REPORTING VIOLATIONS

### 6.1 Report Abuse

If you witness AUP violations:
- **Email**: abuse@thegavl.com
- **Include**: Date, time, username, description, evidence (logs, screenshots)
- **Response time**: We investigate all reports within 24 hours

### 6.2 Law Enforcement Cooperation

We cooperate fully with law enforcement:
- We preserve logs and user data for 10 years
- We respond to subpoenas, court orders, and search warrants
- We may proactively report suspected criminal activity
- We provide testimony in legal proceedings when required

### 6.3 Zero Tolerance

**Severe violations result in:**
- Immediate account suspension
- Reporting to law enforcement
- Civil and criminal prosecution
- Permanent ban from all TheGAVL services
- Public disclosure of violations (if legally permissible)

---

## 7. ENFORCEMENT

### 7.1 Monitoring & Auditing

We monitor Tool usage for compliance:
- Automated detection of suspicious patterns
- Manual review of high-risk activities
- Logging of all scan targets and results
- Periodic security audits of user accounts

### 7.2 Account Suspension

We may suspend accounts for:
- Suspected AUP violations (pending investigation)
- Scanning prohibited targets
- Excessive scanning volume or rate
- Unverified professional credentials
- Law enforcement requests

### 7.3 Account Termination

Permanent termination occurs for:
- Confirmed AUP violations
- Criminal activity using the Tools
- Repeated warnings or temporary suspensions
- False information during registration
- Sharing accounts or credentials

---

## 8. CONSEQUENCES OF VIOLATIONS

### 8.1 Legal Consequences

**Unauthorized use of the Tools may result in:**
- **Criminal charges**: Hacking charges under CFAA (up to 10-20 years imprisonment)
- **Civil liability**: Lawsuits from affected parties for damages
- **Fines**: Up to $250,000 per violation for CFAA violations
- **Restitution**: Court-ordered repayment for damages caused
- **Permanent criminal record**: Affecting employment and travel

### 8.2 Professional Consequences

Violations may result in:
- Revocation of security certifications (CEH, OSCP, etc.)
- Termination from employment
- Blacklisting from the security community
- Loss of government clearance
- Inability to obtain cyber liability insurance

### 8.3 Financial Consequences

You may be held liable for:
- Damages caused to target systems
- Legal fees incurred by affected parties
- Remediation costs
- Business interruption losses
- Regulatory fines incurred by the target

---

## 9. BEST PRACTICES

### 9.1 Pre-Engagement Checklist

Before using the Tools:
- [ ] Obtain written authorization
- [ ] Define clear scope and rules of engagement
- [ ] Verify insurance coverage
- [ ] Establish emergency contact procedures
- [ ] Coordinate with client IT/security teams
- [ ] Document all authorizations
- [ ] Review applicable laws and regulations

### 9.2 During Engagement

While testing:
- [ ] Maintain detailed activity logs
- [ ] Stay within defined scope
- [ ] Report issues immediately to client
- [ ] Stop if you encounter unexpected behavior
- [ ] Communicate regularly with client
- [ ] Validate findings before reporting

### 9.3 Post-Engagement

After testing:
- [ ] Provide comprehensive report to client
- [ ] Delete all data obtained from target systems
- [ ] Follow agreed-upon disclosure timeline
- [ ] Retain logs for legal/compliance purposes
- [ ] Debrief with client
- [ ] Archive all engagement documentation

---

## 10. EXAMPLES

### 10.1 Acceptable Use Examples

✅ **ALLOWED:**
- **Example 1**: You own an e-commerce website. You use the Tools to scan your own site for vulnerabilities before launch.
- **Example 2**: A client hires you for penetration testing and provides a signed SOW defining scope. You test only the systems listed in the SOW during agreed time windows.
- **Example 3**: You participate in a bug bounty program on HackerOne. You comply with the program's rules and use the Tools to find vulnerabilities.
- **Example 4**: You're a student learning security in a lab environment with vulnerable VMs (e.g., Metasploitable, DVWA). You test these isolated systems.

### 10.2 Prohibited Use Examples

❌ **NOT ALLOWED:**
- **Example 1**: You scan random internet hosts to "see what's out there." (Illegal: Unauthorized access)
- **Example 2**: You discover a vulnerability in a website you don't own and demand payment to disclose it. (Illegal: Extortion)
- **Example 3**: During authorized testing, you discover an adjacent network segment and scan it without additional authorization. (Violation: Scope creep)
- **Example 4**: You use the Tools to test a competitor's website to gain business advantage. (Illegal: Corporate espionage)
- **Example 5**: You find credentials during testing and use them to access user accounts. (Illegal: Unauthorized access to stored communications)

---

## 11. USER RESPONSIBILITIES

By using the Tools, you affirm that:
- You have read and understand this AUP
- You will use the Tools only for authorized, lawful purposes
- You have obtained all necessary permissions before testing
- You understand the legal and ethical obligations of security testing
- You accept full responsibility for your use of the Tools
- You will indemnify TheGAVL for any claims arising from your misuse

---

## 12. UPDATES TO THIS POLICY

We may update this AUP at any time:
- Changes effective immediately for new users
- Existing users notified via email 30 days before material changes
- Continued use after changes constitutes acceptance

---

## 13. QUESTIONS & SUPPORT

For questions about this AUP:
- **General inquiries**: support@thegavl.com
- **Legal questions**: legal@thegavl.com
- **Abuse reports**: abuse@thegavl.com
- **Law enforcement**: lawenforcement@thegavl.com (24/7)

---

## 14. ACKNOWLEDGMENT

**BY USING THE TOOLS, YOU ACKNOWLEDGE:**
- You have read and understand this Acceptable Use Policy
- You agree to comply with all provisions of this AUP
- You understand that violations may result in legal consequences
- You will use the Tools responsibly and ethically

---

**Last Updated:** October 16, 2025
**Version:** 1.0.0

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
