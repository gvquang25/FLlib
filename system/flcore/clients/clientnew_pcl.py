"""
Backward compatibility wrapper for clientNewPCL -> clientSCD.
"""
from flcore.clients.clientscd import (
    clientSCD,
    ClientSCD,
    clientNewPCL,
    ClientNewPCL,
)

__all__ = ["clientSCD", "ClientSCD", "clientNewPCL", "ClientNewPCL"]
