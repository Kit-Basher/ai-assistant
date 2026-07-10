from __future__ import annotations

import unittest

from agent.capability_policy import (
    POLICY_SCHEMA_VERSION,
    CapabilityDefinition,
    CapabilityRegistry,
    TrustedInvocationContext,
    authorize_capability,
    build_default_capability_registry,
    stable_fingerprint,
    validate_trusted_invocation_context,
)


class CapabilityPolicyTests(unittest.TestCase):
    def test_registry_loads_migrated_capabilities(self) -> None:
        registry = build_default_capability_registry()
        self.assertEqual("plan_and_confirm", registry.get("system.package.install").authorization_mode)
        self.assertEqual("local_activation_and_confirm", registry.get("system.uninstall").authorization_mode)
        self.assertEqual("allow", registry.get("system.package.inspect").authorization_mode)

    def test_schema_rejects_invalid_enum_and_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid_capability_id"):
            CapabilityDefinition(
                schema_version=POLICY_SCHEMA_VERSION,
                capability_id="System.Install",
                title="bad",
                effect="mutating",
                scope="local_host",
                reversibility="reversible",
                risk_level="medium",
                authorization_mode="plan_and_confirm",
                receipt_required=True,
                runtime_revalidation_required=True,
                target_binding_required=True,
                external_side_effect=False,
                generic_bypass_forbidden=True,
            ).validate()
        with self.assertRaisesRegex(ValueError, "invalid_effect"):
            CapabilityDefinition(
                schema_version=POLICY_SCHEMA_VERSION,
                capability_id="system.bad",
                title="bad",
                effect="maybe",
                scope="local_host",
                reversibility="reversible",
                risk_level="medium",
                authorization_mode="plan_and_confirm",
                receipt_required=True,
                runtime_revalidation_required=True,
                target_binding_required=True,
                external_side_effect=False,
                generic_bypass_forbidden=True,
            ).validate()

    def test_duplicate_registration_rejected(self) -> None:
        registry = CapabilityRegistry()
        definition = build_default_capability_registry().get("system.package.inspect")
        registry.register(definition)
        with self.assertRaisesRegex(ValueError, "duplicate_capability_id"):
            registry.register(definition)

    def test_critical_capability_with_weak_auth_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "critical_capability_requires_activation_or_deny"):
            CapabilityDefinition(
                schema_version=POLICY_SCHEMA_VERSION,
                capability_id="system.critical",
                title="critical",
                effect="mutating",
                scope="local_host",
                reversibility="irreversible",
                risk_level="critical",
                authorization_mode="plan_and_confirm",
                receipt_required=True,
                runtime_revalidation_required=True,
                target_binding_required=True,
                external_side_effect=False,
                generic_bypass_forbidden=True,
            ).validate()

    def test_authorization_decisions(self) -> None:
        read_only = authorize_capability("system.package.inspect")
        self.assertTrue(read_only.allowed)
        self.assertEqual("read_only_allowed", read_only.reason_code)

        install_without_plan = authorize_capability("system.package.install")
        self.assertFalse(install_without_plan.allowed)
        self.assertEqual("plan_required", install_without_plan.reason_code)

        target_fp = stable_fingerprint({"package": "htop"})
        plan_fp = stable_fingerprint({"plan": "install-htop"})
        install_without_confirmation = authorize_capability(
            "system.package.install",
            target_snapshot={"target_fingerprint": target_fp},
            plan_context={"plan_fingerprint": plan_fp, "target_fingerprint": target_fp, "policy_version": POLICY_SCHEMA_VERSION},
        )
        self.assertFalse(install_without_confirmation.allowed)
        self.assertEqual("confirmation_required", install_without_confirmation.reason_code)

        allowed = authorize_capability(
            "system.package.install",
            target_snapshot={"target_fingerprint": target_fp},
            plan_context={"plan_fingerprint": plan_fp, "target_fingerprint": target_fp, "policy_version": POLICY_SCHEMA_VERSION},
            confirmation_context={"confirmed": True},
        )
        self.assertTrue(allowed.allowed)
        self.assertTrue(allowed.mutation_allowed)

    def test_stale_changed_policy_and_target_block(self) -> None:
        target_fp = stable_fingerprint({"package": "htop"})
        plan_fp = stable_fingerprint({"plan": "install-htop"})
        changed = authorize_capability(
            "system.package.install",
            target_snapshot={"target_fingerprint": stable_fingerprint({"package": "curl"})},
            plan_context={"plan_fingerprint": plan_fp, "target_fingerprint": target_fp, "policy_version": POLICY_SCHEMA_VERSION},
            confirmation_context={"confirmed": True},
        )
        self.assertFalse(changed.allowed)
        self.assertEqual("target_changed", changed.reason_code)

        policy_changed = authorize_capability(
            "system.package.install",
            target_snapshot={"target_fingerprint": target_fp},
            plan_context={"plan_fingerprint": plan_fp, "target_fingerprint": target_fp, "policy_version": 999},
            confirmation_context={"confirmed": True},
        )
        self.assertFalse(policy_changed.allowed)
        self.assertEqual("policy_changed", policy_changed.reason_code)

    def test_uninstall_requires_activation(self) -> None:
        decision = authorize_capability(
            "system.uninstall",
            target_snapshot={"target_fingerprint": "target"},
            plan_context={"plan_fingerprint": "plan", "target_fingerprint": "target", "policy_version": POLICY_SCHEMA_VERSION},
            confirmation_context={"confirmed": True},
            activation_context={"valid": False, "reason_code": "local_activation_required"},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual("local_activation_required", decision.reason_code)

    def test_invocation_context_validation(self) -> None:
        ctx = TrustedInvocationContext(
            capability_id="system.package.install",
            executor_id="shell.install_package.v1",
            authorization_decision_id="authz-abc123",
            plan_fingerprint="plan",
            operation_id="op",
        ).to_dict()
        ok, reason, _parsed = validate_trusted_invocation_context(
            ctx,
            capability_id="system.package.install",
            executor_id="shell.install_package.v1",
            plan_fingerprint="plan",
        )
        self.assertTrue(ok)
        self.assertEqual("allowed", reason)
        ok, reason, _parsed = validate_trusted_invocation_context(
            {**ctx, "executor_id": "forged.executor"},
            capability_id="system.package.install",
            executor_id="shell.install_package.v1",
            plan_fingerprint="plan",
        )
        self.assertFalse(ok)
        self.assertEqual("executor_mismatch", reason)


if __name__ == "__main__":
    unittest.main()
