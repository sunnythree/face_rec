package com.example.myapplication;

import android.graphics.PixelFormat;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;

import java.io.IOException;
import java.util.List;

public class MCamera {
    private static Camera mCamera;
    private static MCamera instance=null;
    private String Tag="IrisCameraManager";
    private CameraCallBack cameraCallBack = new CameraCallBack();
    public static MCamera getInstance() {
        if(instance == null) {
            synchronized(MCamera.class){
                if(instance == null) {
                    instance = new MCamera();
                }
            }
        }
        return instance;
    }

    private MCamera() {
    }

    public void startPreview(SurfaceHolder holder) {
        if(null== mCamera) {
            try {
                mCamera = Camera.open(1);
                mCamera.setPreviewDisplay(holder);
                Camera.Parameters parameters = mCamera.getParameters();
                parameters.setPreviewFormat(PixelFormat.YCbCr_420_SP);
                List<Camera.Size> previewSizes = mCamera.getParameters().getSupportedPreviewSizes();
                for (int i=0; i<previewSizes.size(); i++) {
                    Camera.Size pSize = previewSizes.get(i);
                    Log.i(Tag+"--------initCamera", "--------------------previewSize.width = "+pSize.width+"-----------------previewSize.height = "+pSize.height);
                }

                parameters.setPreviewSize(1280, 960);
                parameters.set("orientation", "portrait"); //
                mCamera.setParameters(parameters);
                mCamera.setDisplayOrientation(90);
                mCamera.setPreviewCallback(cameraCallBack);
                mCamera.setFaceDetectionListener(cameraCallBack);
                mCamera.startPreview();
                mCamera.startFaceDetection();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }else {
            mCamera.startPreview();
        }
    }


    public void releaseCamera() {
        if(mCamera!=null) {
            Log.i(Tag,"ThomasYang Stop View finish");
            mCamera.release();
            mCamera=null;
        }
    }
}
