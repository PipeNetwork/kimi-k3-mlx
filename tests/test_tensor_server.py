import argparse
import socket
import threading
import unittest

from scripts.tensor_server import (
    FixtureTokenizer,
    JsonChannel,
    output_digest,
    parse_endpoint,
)
from scripts.benchmark_server import parse_concurrencies


class TestTensorServer(unittest.TestCase):
    def test_control_protocol_round_trip(self):
        left, right = socket.socketpair()
        sender, receiver = JsonChannel(left), JsonChannel(right)
        payload = {"op": "generate", "tokens": list(range(1000))}
        thread = threading.Thread(target=sender.send, args=(payload,))
        thread.start()
        self.assertEqual(receiver.receive(), payload)
        thread.join()
        sender.close()
        receiver.close()

    def test_endpoint_parser(self):
        self.assertEqual(parse_endpoint("10.0.0.1:1234"), ("10.0.0.1", 1234))
        with self.assertRaises(ValueError):
            parse_endpoint("missing-port")

    def test_output_digest_covers_finish_reason(self):
        first = output_digest([[1, 2]], ["length"])
        self.assertEqual(first, output_digest([[1, 2]], ["length"]))
        self.assertNotEqual(first, output_digest([[1, 2]], ["stop"]))

    def test_fixture_tokenizer_is_bounded(self):
        tokenizer = FixtureTokenizer()
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=True,
            add_generation_prompt=True,
        )
        self.assertTrue(tokens)
        self.assertTrue(all(1 <= token <= 126 for token in tokens))

    def test_benchmark_concurrencies(self):
        self.assertEqual(parse_concurrencies("1,4,16"), [1, 4, 16])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_concurrencies("1,1")


if __name__ == "__main__":
    unittest.main()
