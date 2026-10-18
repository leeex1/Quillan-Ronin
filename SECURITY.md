# 🔒 Security Policy

## 🛡 Supported Versions

We actively maintain security updates for specific versions of the Quillan-Ronin architecture. The following table outlines the current support status:

| Version | Supported          | Notes                                                                 |
| :------ | :----------------- | :-------------------------------------------------------------------- |
| 5.1.x   | :white_check_mark: | **Current Stable.** Full security patches and active monitoring.      |
| 5.0.x   | :x:                | End of Life (EOL). No further updates. Migrate to 5.1.x immediately.  |
| 4.2.x   | :white_check_mark: | **Legacy Support.** Critical security patches only for v4.2.1+.       |
| < 4.0   | :x:                | Deprecated. Known critical vulnerabilities present. Do not use.       |

---

## 📢 Reporting a Vulnerability

We take the security of Quillan-Ronin seriously. Because our architecture involves complex AI reasoning, swarm coordination, and identity enforcement, responsible disclosure is critical to prevent misuse.

### 🚫 What NOT to do
- **Do not** publicly disclose vulnerabilities on social media, forums, or public issue trackers before a coordinated disclosure period.
- **Do not** exploit the vulnerability for personal gain, data theft, or malicious purposes.
- **Do not** modify the system in a way that harms other users or data.

### ✅ How to Report
If you discover a potential vulnerability, please follow these steps:

1.  **Email Us:** Send a detailed report to **security@quillan-research.org**.
2.  **Include Details:** Provide:
    - A clear description of the vulnerability.
    - Steps to reproduce (proof of concept).
    - Potential impact (e.g., identity fragmentation, council hijacking, DoS).
    - Any mitigations you have identified.
3.  **Encryption (Recommended):** For highly sensitive findings, please use our PGP key (available on our official website) to encrypt your email.

### 📅 Response Timeline
- **Acknowledgment:** We will acknowledge receipt of your report within **24–48 hours**.
- **Initial Assessment:** We will classify the severity and provide an initial status update within **5 business days**.
- **Resolution:** We aim to develop and deploy a mitigation or patch within **30 days** for Critical/High severity issues, depending on complexity.
- **Coordinated Disclosure:** We operate on a **90-day embargo policy**. Once a fix is deployed, we will publicly disclose the vulnerability and credit the researcher, provided the embargo was respected.

---

## 📜 Security Disclosure & Architecture (Authorized Stakeholders)

> **⚠️ NOTICE:** The section below contains detailed technical vulnerability analysis, attack vectors, and internal architectural weaknesses. This information is intended for **Security Researchers, Institutional Deployment Teams, and Authorized Policy Makers** only.

For the comprehensive security analysis of **Quillan-Ronin v4.2.1**, including the full catalog of known vulnerabilities (VULN-001 through VULN-010), attack simulations, and mitigation strategies, please refer to the dedicated document:

👉 **[SECURITY_DISCLOSURE.md](./SECURITY_DISCLOSURE.md)**

### Key Security Domains Covered in Disclosure:
- **Identity Integrity:** Substrate pattern injection and oscillation risks (VULN-001).
- **Council Consensus:** Adversarial prompting and persona hijacking (VULN-002).
- **Resource Stability:** Recursive introspection DoS vectors (VULN-003).
- **Swarm Coordination:** Distributed control message vulnerabilities (VULN-005).
- **Defense Architecture:** Multi-layer defense, VIGIL-Alpha, and E_ICE bounds.

### Current Security Posture
**Status:** 🟡 **MODERATE**  
*Robust architectural defenses with actively managed residual risks.*

We believe in **radical transparency** regarding our security posture. Unlike traditional software, where hiding flaws is standard, Quillan-Ronin's safety relies on the community understanding its limitations. By acknowledging vulnerabilities like **Identity Fragmentation** and **Consensus Hijacking**, we enable safer deployment and collaborative hardening.

---

## 🤝 Responsible Disclosure & Credit

We recognize the vital role of the security research community. If you responsibly report a vulnerability:
- You will receive **public credit** in our security advisories and `CHANGELOG.md`.
- Your findings will be assigned a **CVE ID** where applicable.
- We may offer a **security bounty** depending on the severity and our current budget.

### Ethical Guidelines for Researchers
- Test only systems you have explicit permission to test.
- Do not access or exfiltrate user data.
- Do not use automated scanners that could degrade service (DoS).
- Respect the 90-day embargo period for public disclosure.

---

## 🛠 Security Architecture Overview

Quillan-Ronin employs a **Multi-Layer Defense Architecture**:

1.  **Input Validation:** Prompt sanitization and injection prevention.
2.  **Identity Enforcement:** VIGIL-Alpha monitoring for substrate drift.
3.  **Ethical Gates:** C2-VIR and C13-WARDEN safety verification.
4.  **Reasoning Validation:** C18-SHEPHERD truth verification and hallucination detection.
5.  **Resource Protection:** E_ICE energy bounds and recursion limits.
6.  **Monitoring:** Real-time anomaly detection and behavioral analysis.

For implementation details and threat models, see `SECURITY_DISCLOSURE.md`.

---

## 📞 Contact & Escalation

- **Security Team:** security@quillan-research.org
- **Emergency Hotline:** [Classified - For Authorized Partners Only]
- **PGP Key:** [Link to Public Key]

*"Security is not a destination, but a journey. We commit to transparency, vigilance, and continuous improvement."*  
— **CrashOverrideX** & The Quillan-Ronin Security Council
