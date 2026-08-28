# Security Policy

Kestrel-Q is pre-alpha.

Do not rely on it to safely process untrusted model artifacts, prompts, tool calls or persistent session files until the relevant security boundaries are explicitly documented and tested.

## Reporting

Before public launch, configure a private security-reporting channel in the GitHub repository and update this file with the final procedure.

Do not disclose exploitable vulnerabilities publicly before maintainers have had a reasonable opportunity to assess them.

## Areas requiring particular care

- model-file bounds checking
- integer overflow in tensor sizes/offsets
- memory-mapped file validation
- persistent state parsing
- HTTP request parsing
- agent command execution
- path traversal
- unsafe shell construction
