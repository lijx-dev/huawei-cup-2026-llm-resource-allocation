"""仅含人工构造的数学微型样例，不作为正式数据。"""
import csv
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import tempfile
import unittest

from q1.quality.domain_quality import summarize_domain
from q1.quality.entropy import entropy_spearman_weights, spearman_matrix
from q1.quality.normalization import fit_minmax, load_parameters, save_parameters, transform_minmax
from q1.quality.preprocessing import (SchemaNotReadyError, ordered_probability_expectation,
                                      require_schema_ready, scalarize, stable_softmax)
from q1.quality.run import audit_a1_domains, iter_audited_records, load_references
from q1.quality.scoring import score_row, topsis_closeness


class QualityTests(unittest.TestCase):
    def test_ordered_logits_and_numerical_stability(self):
        self.assertAlmostEqual(scalarize([0., 0., 0.], {"name":"grade","scalarization":"ordered_logits_expectation","expected_length":3}),0.5)
        self.assertAlmostEqual(scalarize([1000., 1001.], {"name":"grade","scalarization":"ordered_logits_expectation","expected_length":2}),1/(1+math.exp(-1)))
        self.assertAlmostEqual(sum(stable_softmax([1000.,1001.])),1)

    def test_probabilities_not_softmaxed_twice(self):
        self.assertAlmostEqual(ordered_probability_expectation([0.1,0.9]),0.9)
        self.assertRaises(ValueError,ordered_probability_expectation,[0.1,0.8])

    def test_list_semantics_and_invalid(self):
        self.assertEqual(scalarize([1.,2.,3.,4.],{"name":"qurater","scalarization":"list_element_index_3","expected_length":4}),4.)
        self.assertEqual(scalarize([1.,2.,3.,4.],{"name":"qurater","scalarization":"list_equal_mean_four","expected_length":4}),2.5)
        self.assertEqual(scalarize(10000,{"name":"word_count","scalarization":"interval_desirability","interval":[50,10000]}),1)
        self.assertAlmostEqual(scalarize(25,{"name":"word_count","scalarization":"interval_desirability","interval":[50,10000]}),0.5)
        self.assertRaises(ValueError,scalarize,[1.,float('nan')],{"name":"binary","scalarization":"binary_logits_probability_index_1","expected_length":2})
        self.assertRaises(ValueError,scalarize,[1.],{"name":"grade","scalarization":"ordered_logits_expectation","expected_length":2})

    def test_schema_blocks_unconfirmed(self):
        schema={"indicators":[{"name":"x","raw_type":"list","scalarization":None,"direction":None,"decision":"unknown"}]}
        self.assertRaises(SchemaNotReadyError,require_schema_ready,schema,[],["x"])

    def test_fixed_minmax_outside_and_reload(self):
        parameters=fit_minmax([[0.,1.],[10.,3.]],["positive","negative"])
        z,flags=transform_minmax([5.,2.],parameters)
        self.assertEqual(z,[0.5,0.5]);self.assertEqual(flags,[False,False])
        z,flags=transform_minmax([20.,0.],parameters)
        self.assertEqual(z,[1.,1.]);self.assertEqual(flags,[True,True])
        self.assertRaises(ValueError,transform_minmax,[20.,0.],parameters,"reject_out_of_reference")
        self.assertRaises(ValueError,fit_minmax,[[1.,1.],[1.,2.]],["positive","negative"])
        self.assertRaises(ValueError,fit_minmax,[[0.,0.],[float('nan'),1.]],["positive","positive"])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"params.json";save_parameters(path,parameters)
            self.assertEqual(load_parameters(path),parameters)

    def test_entropy_spearman_and_scores(self):
        matrix=[[0.,0.,1.],[0.5,1.,0.],[1.,0.5,0.5],[0.25,0.2,0.8]]
        result=entropy_spearman_weights(matrix)
        self.assertTrue(all(0<=e<=1 for e in result["entropy"]))
        self.assertTrue(all(math.isclose(d,1-e) for d,e in zip(result["difference"],result["entropy"])))
        self.assertTrue(math.isclose(sum(result["weights"]),1))
        for j in range(3):
            self.assertTrue(math.isclose(result["information"][j],result["difference"][j]*result["independence"][j]))
            for k in range(3):
                self.assertAlmostEqual(result["spearman"][j][k],result["spearman"][k][j])
                self.assertTrue(-1<=result["spearman"][j][k]<=1)
        weights=result["weights"]
        for row in matrix:
            score=score_row(row,weights)
            self.assertAlmostEqual(score["Q"],100*sum(x*w for x,w in zip(row,weights)))
            self.assertTrue(0<=score["Q"]<=100)
            self.assertAlmostEqual(score["q"],score["Q"]/100)
            self.assertTrue(0<=topsis_closeness(row,weights)<=1)
        self.assertRaises(ValueError,entropy_spearman_weights,[[0.,1.],[0.,0.],[0.,0.5]])

    def test_domain_one_record_and_provenance(self):
        self.assertEqual(summarize_domain([40.])["q25_Q"],40.)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/"a.jsonl.xz";raw=b'{"id":"x","value":1}\n'
            with lzma.open(path,"wb") as stream:stream.write(raw)
            view=root/"view.csv.gz"
            with gzip.open(view,"wt",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=["source_file","attachment","line_number","record_id","record_hash","status"])
                writer.writeheader();writer.writerow({"source_file":"a.jsonl.xz","attachment":"A1","line_number":1,"record_id":"x","record_hash":hashlib.sha256(raw).hexdigest(),"status":"valid"})
            refs,counts=load_references(view)
            self.assertEqual(counts[("A1","valid")],1)
            rows=list(iter_audited_records(path,root,refs))
            self.assertEqual(rows[0][0]["id"],"x")

    def test_a1_domain_exclusions_preserve_unparseable_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/"a.jsonl.xz";view=root/"view.csv.gz"
            with lzma.open(path,"wb") as stream:
                stream.write(b'{"_source_domain":"arxiv","id":"a"}\n')
                stream.write(b'{broken json\n')
            with gzip.open(view,"wt",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=["source_file","attachment","line_number","status"])
                writer.writeheader()
                writer.writerow({"source_file":"a.jsonl.xz","attachment":"A1","line_number":1,"status":"valid"})
                writer.writerow({"source_file":"a.jsonl.xz","attachment":"A1","line_number":2,"status":"rejected_corrupt"})
            counts=audit_a1_domains(path,root,view)
            self.assertEqual(counts[("arxiv","valid")],1)
            self.assertEqual(counts[("unknown","rejected_corrupt")],1)


if __name__ == "__main__":
    unittest.main()
