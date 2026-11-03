"""Exercise the local HTTP API and all four real backend commands."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import demo_server as demo


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = demo.DemoServer(('127.0.0.1', 0), demo.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        return urlopen(Request(self.url + path, data=data, headers=headers or {}), timeout=120)

    def test_status(self):
        with self.request('/api/status') as result:
            self.assertTrue(all(json.load(result)['backends'].values()))

    def test_duplicate_server_rejected(self):
        with self.assertRaises(OSError):
            demo.DemoServer(('127.0.0.1', self.server.server_port), demo.Handler)

    def test_reject_invalid_and_cross_origin(self):
        for payload in ({'samples': -1}, {'workers': True}, {'samples': '10000'}, {'command': 'bad'}):
            with self.assertRaises(HTTPError) as caught:
                self.request('/api/run', payload)
            self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/run', {}, {'Origin': 'https://example.com'})
        self.assertEqual(caught.exception.code, 403)

    def test_busy(self):
        with demo.RUN_LOCK:
            with self.assertRaises(HTTPError) as caught:
                self.request('/api/run', {})
            self.assertEqual(caught.exception.code, 409)

    def test_real_comparison(self):
        with self.request('/api/run', {'samples': 10000, 'runs': 3, 'workers': 2}) as response:
            result = json.load(response)
        rows = result['results']
        self.assertEqual([r['backend'] for r in rows], ['sequential', 'openmp', 'mpi', 'cuda'])
        self.assertEqual(rows[0]['estimate'], rows[1]['estimate'])
        self.assertEqual(rows[0]['estimate'], rows[2]['estimate'])
        self.assertEqual(rows[0]['speedup'], 1)
        for row in rows:
            self.assertEqual(len(row['rows']), 3)
            self.assertGreater(row['median_ms'], 0)
            self.assertAlmostEqual(row['samples_per_sec'], 10000/(row['median_ms']/1000))
            self.assertGreaterEqual(row['runtime_std_ms'], 0)
            self.assertGreaterEqual(row['estimator_std'], 0)
            self.assertTrue(0 <= row['coverage_count'] <= 3)
            self.assertEqual(row['kernel_median_ms'] is not None, row['backend'] == 'cuda')
            self.assertAlmostEqual(row['speedup'], rows[0]['median_ms']/row['median_ms'])


if __name__ == '__main__':
    unittest.main()
