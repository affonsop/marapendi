from marapendi.flow_channels import FlowChannel


def test_flow_channel_post_init():
    ch = FlowChannel(width=1e-3, height=1e-3, n_parallel=14)
    assert ch.hydraulic_diameter == 2 * ch.width * ch.height / (ch.width + ch.height)
    assert ch.channel_flow_section == ch.width * ch.height
    assert ch.total_flow_section == ch.n_parallel * ch.channel_flow_section
