"""SYS command handlers, sharded by domain.

executor_system.py used to be a 1267-LOC class with ~45 handler methods
on one body. Now SystemExecutor composes six mixin modules here, each
self-registering its handlers via @handles_sys(AstType).
"""

from graphstore.dsl.sys._registry import SYS_DISPATCH, handles_sys

# Importing these modules registers every @handles_sys decorator in them.
from graphstore.dsl.sys.queries import SysQueryHandlers
from graphstore.dsl.sys.schema import SysSchemaHandlers
from graphstore.dsl.sys.lifecycle import SysLifecycleHandlers
from graphstore.dsl.sys.pipeline import SysPipelineHandlers
from graphstore.dsl.sys.cron import SysCronHandlers
from graphstore.dsl.sys.evolve import SysEvolveHandlers

__all__ = [
    "SYS_DISPATCH",
    "handles_sys",
    "SysQueryHandlers",
    "SysSchemaHandlers",
    "SysLifecycleHandlers",
    "SysPipelineHandlers",
    "SysCronHandlers",
    "SysEvolveHandlers",
]
