"""The serving job's port check matches how uvicorn binds (regression)."""

import socket

from deploy.serve import port_free


def test_time_wait_connections_do_not_block_a_restart() -> None:
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen()
    assert not port_free(port)  # a listener makes it busy
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = server.accept()
    conn.close()  # the server side closes first: TIME_WAIT on this port
    client.close()
    server.close()
    assert port_free(port)
