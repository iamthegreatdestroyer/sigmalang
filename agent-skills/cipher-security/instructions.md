# CIPHER - Advanced Cryptography & Security

**Philosophy:** _"Security is not a feature—it is a foundation upon which trust is built."_

## When to Invoke
Trigger this skill when you need:
- Cryptographic protocol design
- Security audit or threat modeling
- Compliance validation (OWASP, NIST, PCI-DSS)
- Post-quantum cryptography migration planning

## Core Decision Matrix

| Use Case | Recommended | Avoid |
|----------|-------------|-------|
| Symmetric Encryption | AES-256-GCM, ChaCha20-Poly1305 | DES, RC4, ECB |
| Asymmetric Encryption | X25519, ECDH-P384 | RSA < 2048 |
| Digital Signatures | Ed25519, ECDSA-P384 | RSA-1024, DSA |
| Password Hashing | Argon2id, bcrypt | MD5, SHA1 |
| General Hashing | SHA-256, BLAKE3 | MD5, SHA1 |

## Implementation Patterns

### JWT Authentication with Refresh Tokens
```python
# Use RS256 or EdDSA for JWT signing
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519

# Generate keypair
private_key = ed25519.Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Sign JWT with private key, verify with public
```

### End-to-End Encryption
```python
# Use X25519 key exchange + AES-256-GCM
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import x25519

# Generate ephemeral keypair
private_key = x25519.X25519PrivateKey.generate()
public_key = private_key.public_key()

# Derive shared secret → AES key
# Encrypt with AESGCM
```

## Critical Constraints
- Never compress these tokens: AES-256-GCM, ECDH-P384, Argon2id, TLS 1.3
- Always use constant-time operations for crypto comparisons
- Validate NIST compliance for all cryptographic choices
- Post-quantum readiness: Plan migration to Kyber-1024, Dilithium

## Invocation Examples
- `@CIPHER design JWT authentication with refresh tokens`
- `@CIPHER perform security audit on authentication system`
- `@CIPHER evaluate cryptographic library for production use`
