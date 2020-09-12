package com.example.myapplication;

import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class FaceRec {
    private Module faceRec = null;

    public static float DISTANCE_THRESHOLD = 0.2f;
    public static float DISTANCE_REC_THRESHOLD = 0.1f;

    public FaceRec(){
        faceRec = Module.load(Utils.assetFilePath(Utils.getApplication(), "android_qface.pt"));
    }

    public void L2Norm(float[] data){
        float sum = 0;
        for(int i=0;i<data.length;i++){
            sum += data[i]*data[i];
        }
        float l2 = (float) Math.sqrt(sum);
        for(int i=0;i<data.length;i++){
            data[i] /= l2;
        }
    }

    public float[] forward(Bitmap bitmap){
        long start = System.currentTimeMillis();
        IValue output  = faceRec.forward(IValue.from(TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)));
        Log.d("ttttt", "forward cost: "+(System.currentTimeMillis()-start));
        float[] feature = output.toTensor().getDataAsFloatArray();
        L2Norm(feature);
        return feature;
    }
}
