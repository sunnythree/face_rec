package com.example.myapplication;

import android.graphics.Bitmap;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class FaceRec {
    private Module faceRec = null;
    private float[] mean = new float[]{0.5f, 0.5f, 0.5f};
    private float[] std = new float[]{0.5f, 0.5f, 0.5f};
    public static int DISTANCE_THRESHOLD = 230;

    public FaceRec(){
        faceRec = Module.load(Utils.assetFilePath(Utils.getApplication(), "android_qface.pt"));
    }
    public float[] forward(Bitmap bitmap){
        IValue output  = faceRec.forward(IValue.from(TensorImageUtils.bitmapToFloat32Tensor(bitmap, mean, std)));
        return output.toTensor().getDataAsFloatArray();
    }
}
