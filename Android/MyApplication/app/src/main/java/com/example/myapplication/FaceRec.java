package com.example.myapplication;

import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class FaceRec {
    private Module faceRec = null;

    public static int DISTANCE_THRESHOLD = 150;

    public FaceRec(){
        faceRec = Module.load(Utils.assetFilePath(Utils.getApplication(), "android_qface.pt"));
    }
    public float[] forward(Bitmap bitmap){
        long start = System.currentTimeMillis();
        IValue output  = faceRec.forward(IValue.from(TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)));
        Log.d("ttttt", "forward cost: "+(System.currentTimeMillis()-start));
        return output.toTensor().getDataAsFloatArray();
    }
}
