"""Precomputed public title matrices must be reused safely and fail closed."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from scripts import build_semantic_assets as builder

class PrecomputedAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.cache=self.root/'cache';self.cache.mkdir()
        self.records=[{'code':str(i),'title':str(i)} for i in range(3)]
        self.catalogue=self.root/'mms.json';self.catalogue.write_text(json.dumps(self.records))
        self.vectors=np.eye(3,4,dtype=np.float32)
        np.save(self.cache/'title_embeddings.npy',self.vectors,allow_pickle=False)
        self.meta={'model_id':'test-model','model_revision':'test-revision','pooling':'cls','max_length':64,'dimension':4,'catalogue_sha256':builder.catalogue_digest(self.records)}
        self.save_metadata()

    def save_metadata(self):
        self.meta['embeddings_sha256']=hashlib.sha256((self.cache/'title_embeddings.npy').read_bytes()).hexdigest()
        (self.cache/'metadata.json').write_text(json.dumps(self.meta))

    def load(self):
        return builder.load_precomputed(self.cache,self.records,'test-model','test-revision','cls',64)

    def test_valid_cache(self):self.assertEqual(self.load(),self.meta)

    def test_changed_catalogue_model_or_settings_rejected(self):
        for key in ('catalogue_sha256','model_id','model_revision','pooling','max_length'):
            with self.subTest(key=key):
                original=self.meta[key];self.meta[key]='wrong';self.save_metadata()
                with self.assertRaises(ValueError):self.load()
                self.meta[key]=original;self.save_metadata()

    def test_corruption_shape_and_non_normalized_vectors_rejected(self):
        (self.cache/'title_embeddings.npy').write_bytes(b'broken')
        with self.assertRaises(ValueError):self.load()
        for values in (np.zeros((2,4),dtype=np.float32),np.zeros((3,4),dtype=np.float32),np.ones((3,4),dtype=np.float64)):
            np.save(self.cache/'title_embeddings.npy',values);self.save_metadata()
            with self.assertRaises(ValueError):self.load()

    def test_setup_reuses_exact_bytes_without_encoding(self):
        tokenizer=Mock();model=Mock();model.config=SimpleNamespace(hidden_size=4)
        output=self.root/'runtime'
        args=['build','--catalogue',str(self.catalogue),'--output',str(output),'--model','test-model','--revision','test-revision','--precomputed',str(self.cache)]
        with patch('sys.argv',args),patch.object(builder.AutoTokenizer,'from_pretrained',return_value=tokenizer),patch.object(builder.AutoModel,'from_pretrained',return_value=model),patch.object(builder,'encode',side_effect=AssertionError('must not encode')):
            builder.main()
        self.assertEqual((output/'title_embeddings.npy').read_bytes(),(self.cache/'title_embeddings.npy').read_bytes())
        self.assertEqual((output/'metadata.json').read_bytes(),(self.cache/'metadata.json').read_bytes())
        model.save_pretrained.assert_called_once()

    def test_checked_in_assets_match_public_catalogue_and_pins(self):
        root=Path(__file__).resolve().parents[1]
        records=json.loads((root/'data/mms-2026-01.json').read_text())
        pins={'english':('cambridgeltl/SapBERT-from-PubMedBERT-fulltext','090663c3ae57bf35ffe4d0d468a2a88d03051a4d'),
              'multilingual':('cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR','47b6bd041ba61311584bb2494edfda5c7d9b719f')}
        for language,(model,revision) in pins.items():
            metadata=builder.load_precomputed(root/'assets/catalogue'/language,records,model,revision,'cls',64)
            self.assertEqual(metadata['dimension'],768)
            self.assertEqual(len(records),18375)
