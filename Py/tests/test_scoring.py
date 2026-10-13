"""Synthetic numerical/domain tests; no cohort records or saved scores."""
import gc
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
import pandas as pd
from DISCO4HD import cal_disco
try:
    from DISCO4HD.disco_optimized import disco_optimized_cpp as native
except ImportError:
    native = None

def fixture():
    rng=np.random.default_rng(271828)
    values=rng.normal(size=(57,5))*np.array([.3,2,5,.7,9])+np.arange(5)
    age=np.linspace(19,85,len(values))+rng.normal(size=len(values))
    values += (age-age.mean())[:,None]*np.array([.008,-.035,.05,.02,-.04])
    names=['a','b','c','d','e'];d=pd.DataFrame(values,columns=names);d['age']=age
    return d,d.iloc[:11].copy(),names

def independent(data,reference,weights):
    baseline=np.corrcoef(reference,rowvar=False)
    out=[]
    for x in data:
        perturbed=np.corrcoef(np.vstack([reference,x]),rowvar=False)
        with np.errstate(divide='ignore'):
            out.append(np.log(np.sum((baseline-perturbed)**2*weights)*len(reference)**2))
    return np.array(out)

class ScoringTests(unittest.TestCase):
    def test_numpy_matches_independent_pearson_and_preserves_input(self):
        d,r,n=fixture();original=d.copy(deep=True)
        cc=np.array([np.corrcoef(d['age'],d[v])[0,1] for v in n])
        w=np.abs(np.outer(cc,cc));np.fill_diagonal(w,0);w/=w.sum()
        score=cal_disco(d,n,r,use_cpp=False)
        np.testing.assert_allclose(score.DISCO,independent(d[n].to_numpy(),r[n].to_numpy(),w),rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(d,original)

    @unittest.skipUnless(native is not None,'Optional native extension is absent')
    def test_native_numpy_every_score(self):
        d,r,n=fixture()
        np.testing.assert_allclose(cal_disco(d,n,r,use_cpp=True).DISCO,cal_disco(d,n,r,use_cpp=False).DISCO,rtol=0,atol=5e-11)

    @unittest.skipUnless(native is not None,'Optional native extension is absent')
    def test_C_F_strided_and_nonsymmetric_weight_field_mapping(self):
        d=np.array([[1.,2.,-7.],[3.,11.,4.],[-2.,.1,9.],[4.,-3.,1.]])
        r=np.array([[1.,3.,2.],[-1.,2.,5.],[4.,-.5,11.],[2.,8.,-.3],[-7.,1.,8.]])
        w=np.array([[0.,.11,.02],[.31,0.,.05],[.17,.34,0.]])
        expected=independent(d,r,w)
        for order in ['C','F']:
            np.testing.assert_allclose(native(np.array(d,order=order),np.array(r,order=order),np.array(w,order=order))[:,0],expected,rtol=0,atol=5e-12)
        np.testing.assert_allclose(native(d[:,::-1],r[:,::-1],w[::-1,::-1])[:,0],expected,rtol=0,atol=5e-12)

    @unittest.skipUnless(native is not None,'Optional native extension is absent')
    def test_native_return_memory_owned_after_gc_and_repeated_calls(self):
        d,r,n=fixture();a=d[n].to_numpy();b=r[n].to_numpy();w=np.ones((5,5));np.fill_diagonal(w,0)
        result=native(a,b,w);expected=result.copy()
        self.assertTrue(result.flags['OWNDATA']);self.assertIsNone(result.base)
        del a,b,w;gc.collect()
        for k in range(100):native(d[n].to_numpy()+k*.001,r[n].to_numpy(),np.eye(5))
        gc.collect();np.testing.assert_array_equal(result,expected)

    @unittest.skipUnless(native is not None,'Optional native extension is absent')
    def test_native_dimension_nonfinite_and_variance_guards(self):
        d,r,n=fixture();a=d[n].to_numpy();b=r[n].to_numpy();w=np.ones((5,5))
        cases=[(a[:,0],b,w),(a[:,:4],b,w),(a,b,w[:,:4]),(a,b[:2],w),(a[:,[0]],b[:,[0]],np.ones((1,1))),
               (a,np.ones_like(b),w),(a,b,-w),(a*np.nan,b,w),(a,b,w*np.inf)]
        for x,y,z in cases:
            with self.subTest(shapes=[x.shape,y.shape,z.shape]),self.assertRaises((ValueError,RuntimeError)):native(x,y,z)

    @unittest.skipUnless(native is not None,'Optional native extension is absent')
    def test_zero_perturbation_is_negative_infinity_not_floor(self):
        r=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
        w=np.array([[0.,.5],[.5,0.]])
        self.assertTrue(np.isneginf(native(np.zeros((1,2)),r,w)[0,0]))

    def test_frontend_validation_and_unsupported_numba(self):
        d,r,n=fixture()
        bad=[]
        x=d.copy();x.loc[0,n[0]]=np.nan;bad.append((x,n,r,{}))
        x=d.copy();x['age']=1;bad.append((x,n,r,{}))
        x=d.copy();x['age']=x['age']>40;bad.append((x,n,r,{}))
        x=d.copy();x[n[0]]=x[n[0]]>0;bad.append((x,n,r,{}))
        x=d.copy();x[n[0]]=1;bad.append((x,n,r,{}))
        x=r.copy();x[n[0]]=1;bad.append((d,n,x,{}))
        bad += [(d,[n[0]],r,{}),(d,[n[0],n[0]],r,{}),(d,n,r.iloc[:2],{}),(d,n,r,{'ncores':0}),(d,n,r,{'use_cpp':'yes'}),(d,n+['missing'],r,{})]
        for x,markers,y,kw in bad:
            with self.subTest(kw=kw),self.assertRaises(ValueError):cal_disco(x,markers,y,use_cpp=False,**kw) if 'use_cpp' not in kw else cal_disco(x,markers,y,**kw)
        with self.assertRaises(NotImplementedError):cal_disco(d,n,r,use_numba=True)
        zero=pd.DataFrame({'age':[-1.,0.,1.],'a':[1.,-2.,1.],'b':[2.,-4.,2.]})
        with self.assertRaises(ValueError):cal_disco(zero,['a','b'],zero,use_cpp=False)

    def test_reference_and_analysis_column_permutation_invariance(self):
        d,r,n=fixture();a=cal_disco(d,n,r,use_cpp=False).DISCO
        b=cal_disco(d,list(reversed(n)),r,use_cpp=False).DISCO
        np.testing.assert_allclose(a,b,rtol=0,atol=2e-12)

    def test_independent_R_definition_every_score(self):
        rscript=os.environ.get('DISCO_RSCRIPT') or shutil.which('Rscript')
        if not rscript:self.skipTest('Rscript not available; NumPy/native checks still run')
        package=Path(__file__).resolve().parents[2]
        d,r,n=fixture()
        with tempfile.TemporaryDirectory(prefix='disco_cross_R_') as tmp:
            tmp=Path(tmp);d.to_csv(tmp/'d.csv',index=False);r.to_csv(tmp/'ref.csv',index=False)
            command="args<-commandArgs(TRUE);source(file.path(args[1],'R/validation.R'));source(file.path(args[1],'R/cal_disco.R'));d<-read.csv(args[2]);ref<-read.csv(args[3]);v<-setdiff(names(d),'age');score<-cal_disco(d,v,ref,cpp=FALSE)$DISCO;write.csv(data.frame(DISCO=score),args[4],row.names=FALSE)"
            subprocess.run([rscript,'--vanilla','-e',command,str(package),str(tmp/'d.csv'),str(tmp/'ref.csv'),str(tmp/'scores.csv')],check=True,capture_output=True,text=True)
            expected=pd.read_csv(tmp/'scores.csv').DISCO.to_numpy()
            np.testing.assert_allclose(cal_disco(d,n,r,use_cpp=False).DISCO,expected,rtol=0,atol=5e-11)
            if native is not None:np.testing.assert_allclose(cal_disco(d,n,r,use_cpp=True).DISCO,expected,rtol=0,atol=5e-11)

if __name__=='__main__':unittest.main()
