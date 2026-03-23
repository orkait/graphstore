"""Vault commands are handled by VaultExecutor (separate from the handler registry).

Vault commands are routed via isinstance check in the Executor._dispatch method,
then delegated to VaultExecutor. No @handles registration needed.
"""
