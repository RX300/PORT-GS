"""Conservative Gaussian-source irradiance exchange."""
from .base import PortTransport


def exchange_irradiance(
    source_incident, source_partition, source_exchange_logits, mass,
    incident, log_partition, exchange_logits,
):
    """Exchange Gaussian-source light with continuous material receivers.

    Source-node queries recover the mass-conservative discrete operator.
    Arbitrary receiver locations query the same source integral.
    """
    fraction = source_exchange_logits.sigmoid()
    source_logits = (source_partition + mass.log()[:, None]).T.contiguous()
    source_weights = source_logits.softmax(dim=-1)
    # Factor the channel-dependent normalization into two small port matrices.
    # This computes the same m*a*f average without an N x R x RGB tensor.
    pooled = (source_weights @ (fraction * source_incident)) / (source_weights @ fraction)
    received = log_partition.exp() @ pooled
    fraction = exchange_logits.sigmoid()
    return (1 - fraction) * incident + fraction * received


class AnchorTransport(PortTransport):
    def exchange_radiance(self, source, receiver):
        incident = exchange_irradiance(
            source.incident, self.partition(source.xyz), self.exchange(source.features), source.mass,
            receiver.incident, self.partition(receiver.xyz), self.exchange(receiver.features))
        return receiver.response * incident
