"""Scientific fixture checks for the primary notebook's auxiliary export."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ppg_iot import notebook_workflow as w
from ppg_iot.model import WindowNetwork
from ppg_iot.baselines import fft_hr, ibi_hr

torch.set_num_threads(1)

class NotebookWorkflowTests(unittest.TestCase):
    def test_notebook_has_all_phases_and_small_or_single_function_cells(self):
        import ast
        book=w.read_json(ROOT/"DinhLeHoangDanh_DoAn_PPG_DaLiA.ipynb")
        self.assertTrue(book["metadata"]["ppg_project"]["primary_deliverable"])
        ids=[c["id"] for c in book["cells"]]; self.assertEqual(len(ids),len(set(ids)))
        for cell in book["cells"]:
            if cell["cell_type"]!="code": continue
            source="".join(cell["source"]); tree=ast.parse(source)
            if len(source.splitlines())>35:
                self.fail(f"Code cell {cell.get('id')} exceeds 35 lines")
            if len(source.splitlines())>8:
                functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
                calls=[node for node in tree.body if isinstance(node,(ast.Expr,ast.Assign))]
                self.assertEqual(len(functions),1,source[:120])
                self.assertLessEqual(len(calls),2,source[:120])
                self.assertEqual(len(tree.body),1+len(calls),source[:120])
        text="".join("".join(c["source"]) for c in book["cells"] if c["cell_type"]=="markdown")
        for phase in ("FFT","IBI","Train","Validation","INT8","Test","thất bại","Trình diễn","6,0"):
            self.assertIn(phase,text)

    def test_export_is_exact_notebook_source(self):
        book=w.read_json(ROOT/"DinhLeHoangDanh_DoAn_PPG_DaLiA.ipynb")
        payload='"""Auxiliary export of main notebook scientific cells."""\n' + "\n\n".join(
            "".join(c["source"]) for c in book["cells"]
            if c.get("metadata",{}).get("ppg_project",{}).get("workflow_source"))+"\n"
        self.assertEqual((ROOT/"src/ppg_iot/notebook_workflow.py").read_text(encoding="utf-8"),payload)

    def test_network_matches_original_architecture_and_b1(self):
        for channels,parameters in ((1,25185),(4,25857)):
            model=w.build_network(channels).eval(); original=WindowNetwork(channels).eval()
            original.load_state_dict(model.state_dict())
            self.assertEqual(sum(p.numel() for p in model.parameters()),parameters)
            x=torch.linspace(-2,2,channels*256).reshape(1,channels,256)
            torch.testing.assert_close(model(x),original(x),rtol=0,atol=0)
            self.assertEqual(model(x).shape,(1,1))

    def test_notebook_dsp_matches_independent_baseline_module(self):
        for fs in (32,64):
            t=np.arange(8*fs)/fs
            for bpm in (60,120):
                signal=np.sin(2*np.pi*bpm/60*t)
                self.assertEqual(w.fft_prediction(signal,fs)["hr_pred_bpm"],fft_hr(signal,fs).hr_pred_bpm)
                self.assertAlmostEqual(w.ibi_prediction(signal,fs)["hr_pred_bpm"],ibi_hr(signal,fs).hr_pred_bpm)
        self.assertFalse(w.ibi_prediction(np.zeros(256))["prediction_valid"])
        self.assertTrue(w.fft_prediction(np.zeros(256))["prediction_valid"])

    def test_train_loss_shape_eval_freezes_bn_and_seed_repeat(self):
        outputs=[]
        for _ in range(2):
            gen=w.seed_everything(42); model=w.build_network(4)
            x=torch.randn(5,4,256); y=torch.full((5,1),70.)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            loss=w.train_epoch(model,optimizer,x,y,gen,4)
            self.assertTrue(np.isfinite(loss))
            running=model.bn1.running_mean.clone()
            outputs.append(w.predict_tensor(model,x,batch_size=4))
            torch.testing.assert_close(model.bn1.running_mean,running,rtol=0,atol=0)
            with self.assertRaises(ValueError): w.train_epoch(model,optimizer,x,y[:,0],gen,4)
        np.testing.assert_array_equal(*outputs)

    def test_rng_resume_exact(self):
        gen=w.seed_everything(123); state=w.capture_rng(gen)
        first=(np.random.rand(4),torch.rand(4),torch.randperm(9,generator=gen))
        w.restore_rng(state,gen)
        second=(np.random.rand(4),torch.rand(4),torch.randperm(9,generator=gen))
        np.testing.assert_array_equal(first[0],second[0])
        torch.testing.assert_close(first[1],second[1],rtol=0,atol=0)
        torch.testing.assert_close(first[2],second[2],rtol=0,atol=0)

    def test_fp32_artifact_model_only_roundtrip_and_hash(self):
        model=w.build_network(4).eval(); x=torch.randn(3,4,256)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model_fp32.pt"
            saved=w.save_inference_model(model,path,input_channels=4,quantized=False)
            reload=w.load_inference_model(path,saved["sha256"])
            torch.testing.assert_close(model(x),reload(x),rtol=0,atol=0)
            self.assertEqual(saved["bytes"],path.stat().st_size)
            packet=torch.load(path,weights_only=True)
            self.assertNotIn("optimizer",packet)
            with self.assertRaises(ValueError): w.load_inference_model(path,"wrong")

    def test_largest_remainder_no_replacement_capacity(self):
        self.assertEqual(w.largest_remainder({"a":7,"b":2,"c":1},5),{"a":4,"b":1,"c":0})
        self.assertEqual(w.largest_remainder({"a":1,"b":1},2),{"a":1,"b":1})
        with self.assertRaises(ValueError): w.largest_remainder({"a":1},2)

    def test_timing_contract_rejects_incomplete_measurements(self):
        model=w.build_network(4).eval(); x=torch.zeros(2,4,256)
        with self.assertRaises(ValueError): w.measure_latency(model,x,99,1000)
        with self.assertRaises(ValueError): w.measure_latency(model,x,100,999)

if __name__=="__main__": unittest.main()
