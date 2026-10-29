"""
Backward compatibility wrapper for FedNewPCL -> FedSCD.
"""
from flcore.servers.serverscd import (
    FedSCD,
    FedSCD as FedNewPCL,
    ServerSCD,
    serverSCD,
    ServerNewPCL,
    serverNewPCL,
    proto_aggregation_pcl,
    proto_aggregation_scd,
)

__all__ = [
    "FedSCD",
    "FedNewPCL",
    "ServerSCD",
    "serverSCD",
    "ServerNewPCL",
    "serverNewPCL",
    "proto_aggregation_pcl",
    "proto_aggregation_scd",
]
