---
name: seed
description: Seed test policies (US + India) with valid dates into Postgres
disable-model-invocation: true
allowed-tools: Bash(python *)
---

# Seed Test Policies

Run the policy seeder to populate both US and India test policies into Neon Postgres:

```bash
python scripts/seed_policies.py
```

Report what was seeded (policy count per country). If the backend is running,
mention that new claims can now reference these policies.