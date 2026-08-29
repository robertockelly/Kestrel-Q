# utf8proc provenance

Kestrel-Q vendors utf8proc **2.10.0** from upstream commit
`a1b99daa2a3393884220264c927a48ba1251a9c6`.

Upstream: <https://github.com/JuliaStrings/utf8proc>

Purpose: deterministic UTF-8 validation, Unicode 16.0.0 general-category
lookup and composition primitives for the Qwen3.8-Flash-Next canonical
tokenizer adapter. The pinned oracle uses Unicode 16.0.0 regex properties but
Unicode 9.0.0 NFC tables. Kestrel-Q therefore gates normalization with the
generated Unicode-9 assigned-range table before calling utf8proc composition;
normalization stability makes utf8proc's result identical for code points that
were assigned by Unicode 9, while later code points remain opaque boundaries.
Kestrel-Q does not delegate correctness to Windows or current system tables.

Files retained from upstream:

| File | SHA-256 |
|---|---|
| `utf8proc.c` | `e32cd937c0d7c82b6b499dfb658c8974e6928104b3c4ee3cfcde46799da71aa9` |
| `utf8proc.h` | `71c3c27dc140cc33547f1ec9f5cae267e2ae5dd8d5037bc76847d4f4bdf94e23` |
| `utf8proc_data.c` | `c6e58560e96655fe3beb088517aa954828e60fa1ac7c982240b482d65429f5ea` |
| `LICENSE.md` | `767edbcafb23016d7f203aaf2cbdef5690ad58bcede52ee241427dcb816cd644` |

License: MIT for utf8proc code, with Unicode data-file notices reproduced in
`LICENSE.md`; compatible with Apache-2.0 distribution. The dependency is
compiled as an internal static target and is not exposed by Kestrel-Q's public
API. Upstream warnings are suppressed at the dependency target only; Kestrel-Q
code retains `/W4`.

Alternatives considered were Windows normalization APIs, which would not pin
Unicode semantics independently of the host, and a project-local partial NFC
implementation, which was rejected as incomplete. Removal requires replacing
UTF-8 validation, Unicode category lookup and NFC with an equally pinned,
differentially validated implementation.
