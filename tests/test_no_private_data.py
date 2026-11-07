"""Prevent accidental commits of private inputs or generated clinical output."""
import pathlib,subprocess,unittest,shutil
class RepositoryPrivacyTests(unittest.TestCase):
    def test_no_private_data_is_tracked(self):
        root=pathlib.Path(__file__).resolve().parents[1]
        if not shutil.which('git') or not (root/'.git').exists():self.skipTest('source checkout only')
        files=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
        private=[p for p in files if p.endswith('.jsonl') or p.startswith(('private-data/','reports/','out/')) or (p.startswith('data/') and p!='data/mms-2026-01.json')]
        self.assertEqual(private,[], 'Private data must stay outside Git')
