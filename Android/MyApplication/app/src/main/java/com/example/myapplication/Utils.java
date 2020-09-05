package com.example.myapplication;

import android.app.Application;
import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

public class Utils {

    public static String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e("pytorchandroid", "Error process asset " + assetName + " to file path");
        }
        return null;
    }
    /** 反射获取Application */
    public static Application getApplication() {
        Application application = null;
        try {
            Class localClass1 = Class.forName("com.android.internal.os.RuntimeInit");
            Field localField1 = localClass1.getDeclaredField("mApplicationObject");
            localField1.setAccessible(true);
            Object localObject1 = localField1.get(localClass1);

            Class localClass2 = Class.forName("android.app.ActivityThread$ApplicationThread");
            Field localField2 = localClass2.getDeclaredField("this$0");
            localField2.setAccessible(true);
            Object localObject2 = localField2.get(localObject1);

            Class localClass3 = Class.forName("android.app.ActivityThread");
            Method localMethod = localClass3.getMethod("getApplication", new Class[0]);
            localMethod.setAccessible(true);
            Application localApplication = (Application) localMethod.invoke(localObject2, new Object[0]);
            if (localApplication != null){
                application = localApplication;
            }
        }catch (Exception localException) {
            localException.printStackTrace();
        }

        //Toast.makeText(application, "AppliactionTool -> getApplication()", Toast.LENGTH_SHORT).show();
        return application;
    }

//    欧几里得距离
//    Euclidean distance
    public static float euclideanDistance(float[] featuresA, float[] featuresB){
        if(featuresA.length != featuresB.length){
            return -1f;
        }
        float sum = 0;
        for (int i=0; i< featuresA.length; i++){
            sum += (featuresA[i]-featuresB[i])*(featuresA[i]-featuresB[i]);
        }
        return sum;
    }


}
