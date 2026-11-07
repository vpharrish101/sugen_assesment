"""Optional three-note end-to-end Docker smoke with a local fake Ollama."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,os,subprocess,sys,tempfile,threading,unittest
from pathlib import Path

@unittest.skipUnless(os.getenv('CODER_TEST_NOTES'),'supply external private smoke notes')
class PrototypeDockerTests(unittest.TestCase):
    def test_three_external_notes_actual_http_and_fallback(self):
        root=Path(__file__).resolve().parents[1]
        notes=[json.loads(x) for x in Path(os.environ['CODER_TEST_NOTES']).read_text().splitlines()][:3]
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])));calls.append(payload)
                note=payload['messages'][1]['content'].split('CLINICAL NOTE\n',1)[1].split('\n\nDIAGNOSIS-FOCUS HINT',1)[0]
                value={'choices':[2,1,3]} if payload['format']=='json' else {'choices':['C002','C001','C003'],'evidence':note[:20]}
                content='not JSON' if len(calls)==2 else json.dumps(value)
                response=json.dumps({'message':{'content':content},'done_reason':'stop'}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(response)));self.end_headers();self.wfile.write(response)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                source,output=Path(directory)/'notes.jsonl',Path(directory)/'predictions.jsonl'
                source.write_text(''.join(json.dumps({'id':r['id'],'text':r['text']})+'\n' for r in notes))
                result=subprocess.run([sys.executable,'-m','prototype',str(source),str(output),'--endpoint',f'http://127.0.0.1:{server.server_port}'],cwd='/app',capture_output=True,text=True,timeout=90)
                self.assertEqual(result.returncode,0,result.stderr)
                summary=json.loads(result.stderr.splitlines()[-1]);rows=[json.loads(x) for x in output.read_text().splitlines()]
                valid={r['code'] for r in json.loads(Path('/app/data/mms-2026-01.json').read_text())}
                self.assertEqual([r['id'] for r in rows],[r['id'] for r in notes])
                self.assertTrue(all(len(set(r['codes']))==3 and set(r['codes'])<=valid for r in rows))
                self.assertEqual(len(calls),6 if summary['architecture']=='two-pass' else 3)
                self.assertTrue(summary['semantic_enabled']);self.assertGreater(summary['fallbacks'],0)
        finally:server.shutdown();server.server_close();thread.join(timeout=2)
