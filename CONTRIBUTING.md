# Contributing

Kestrel-Q welcomes contributions once the public repository is opened.

## Before coding

Read:

- `README.md`
- `AGENTS.md`
- `docs/VISION.md`
- `docs/ARCHITECTURE.md`
- relevant ADRs

For non-trivial architectural work, open an issue/design discussion before investing in a large implementation.

## Pull requests

A good PR should:

- solve one coherent problem;
- include tests;
- include benchmark evidence for performance changes;
- update docs when assumptions change;
- avoid unrelated formatting churn;
- explain hardware-specific behavior.

## Performance contributions

Provide raw benchmark data and the complete environment.

A claimed optimization that cannot be reproduced may still be valuable as an experiment, but should not be merged as an established improvement.


## Licensing of contributions

Unless explicitly stated otherwise before submission, contributions intentionally submitted for inclusion in Kestrel-Q are accepted under the **Apache License 2.0**, consistent with Section 5 of that license.

By contributing, you confirm that you have the right to submit the contribution under those terms.

Do not copy source code from projects with licenses that are incompatible with Apache-2.0. When learning from another implementation, preserve clean provenance: understand the algorithm/specification and write an independent implementation unless the source license clearly permits reuse and the required notices are retained.

Third-party dependencies and incorporated source must have their license reviewed before merge.

## Generated code and AI assistance

AI-assisted contributions are allowed.

The contributor remains responsible for:

- correctness;
- licensing;
- provenance;
- tests;
- understanding the submitted code.

Do not submit large AI-generated changes that the contributor cannot explain or maintain.

## Code style

The exact formatter/linter will be selected during Phase 0.

Until then:

- prefer straightforward C;
- explicit ownership;
- bounded operations;
- descriptive names;
- comments for invariants, not narration.
