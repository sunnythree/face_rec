package com.example.myapplication;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.text.TextUtils;
import android.util.JsonReader;
import android.util.Log;
import android.view.Gravity;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.google.gson.Gson;


import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class CameraCallBack implements Camera.PreviewCallback, Camera.FaceDetectionListener {
    private FastYUV2RGB fastYUV2RGB = new FastYUV2RGB(Utils.getApplication());
    private FaceRec faceRec = new FaceRec();
    public static Camera.Face curFace = new Camera.Face();
    private AtomicBoolean getFace = new AtomicBoolean(false);
    private ArrayList<FeaturesWrap> recFeatures = new ArrayList<>();
    private AtomicBoolean msgProcessed = new AtomicBoolean(true);
    private final int USER_EXIST = 3;
    private final int DRAW_FACE = 4;
    private final int CLEAN_FACE = 5;
    private static class FeaturesWrap{
        public FeaturesWrap(int score, float[] features){
            this.score = score;
            this.features = features;
        }
        public int score;
        public float[] features;
    }
    private static class MatchWrap{
        public MatchWrap(float distance, String userName){
            this.distance = distance;
            this.userName = userName;
        }
        public float distance;
        public String userName;
    }
    /**
     * 一个输入框的 dialog
     */
    private void register(final String  features, Bitmap bitmap) {
        final EditText editText = new EditText(MainActivity.context);
        final ImageView imageView = new ImageView(MainActivity.context);
        final LinearLayout linearLayout = new LinearLayout(MainActivity.context);
        imageView.setImageBitmap(bitmap);
        LinearLayout.LayoutParams param = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        param.width = 400;
        param.height = 400;
        param.setMargins(20,20,20,20);
        param.gravity = Gravity.CENTER;
        imageView.setLayoutParams(param);
        linearLayout.addView(imageView);
        LinearLayout.LayoutParams param1 = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        param1.width = 400;
        param1.height = 120;
        param1.gravity = Gravity.CENTER;
        editText.setLayoutParams(param1);
        linearLayout.addView(editText);
        linearLayout.setGravity(Gravity.CENTER_HORIZONTAL);
        linearLayout.setOrientation(LinearLayout.VERTICAL);
        final AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.context).
                setTitle("register new user")
                .setView(linearLayout)
                .setOnCancelListener(new DialogInterface.OnCancelListener() {
                    @Override
                    public void onCancel(DialogInterface dialogInterface) {
                        msgProcessed.set(true);
                    }
                })
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        String userName = editText.getText().toString();
                        if (TextUtils.isEmpty(userName)){
                            userName = "default";
                        }
                        SqliteDb.getInstance().insertOneUser(userName, features);
                        msgProcessed.set(true);
                    }
                });
        builder.create().show();
    }

    private void recognization(String userName) {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.context)
                .setIcon(R.mipmap.ic_launcher)
                .setTitle("Recognization")
                .setMessage(userName)
                .setOnCancelListener(new DialogInterface.OnCancelListener() {
                    @Override
                    public void onCancel(DialogInterface dialogInterface) {
                        msgProcessed.set(true);
                    }
                })
                .setPositiveButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        msgProcessed.set(true);
                    }
                });
        builder.create().show();
    }

    private Handler handler = new Handler(Looper.getMainLooper()){
        @Override
        public void handleMessage(@NonNull Message msg) {
            super.handleMessage(msg);
            switch (msg.what){
                case MainActivity.MODE_RECOGNIZATION:{
                    MainActivity.mode.set(MainActivity.MODE_IDLE);
                    Bundle bundle = msg.getData();
                    String userName = bundle.getString("userName");
                    recognization(userName);
                    break;
                }
                case MainActivity.MODE_REGISTER: {
                    MainActivity.mode.set(MainActivity.MODE_IDLE);
                    Bundle bundle = msg.getData();
                    String features = bundle.getString("features");
                    Bitmap bitmap = bundle.getParcelable("image");
                    register(features, bitmap);
                    break;
                }
                case USER_EXIST:{
                    MainActivity.mode.set(MainActivity.MODE_IDLE);
                    Bundle bundle = msg.getData();
                    String userName = bundle.getString("userName");
                    Toast.makeText(MainActivity.context, "user: "+userName+" existed", Toast.LENGTH_SHORT).show();
                    msgProcessed.set(true);
                    break;
                }
                case DRAW_FACE:{
                    MainActivity.mFaceDraw.draw();
                    break;
                }
                case CLEAN_FACE:{
                    MainActivity.mFaceDraw.clear();
                }
                default:
                    break;
            }
        }
    };

    private int getBestFace(Camera.Face[] faces){
        int maxIndex = 0;
        int maxScore = 0;
        for (int i=0; i<faces.length; i++){
            if(faces[i].score > maxScore){
                maxScore = faces[i].score;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private void normalFace(Camera.Face face, int width, int height){
        if(face.rect.left < 0){
            face.rect.left = 0;
        }
        if(face.rect.top < 0){
            face.rect.top = 0;
        }
        if(face.rect.right > width){
            face.rect.right = width;
        }
        if(face.rect.bottom > height){
            face.rect.bottom = height;
        }
    }

    private void convertFaceIndex(Camera.Face face, int width, int height){
        face.rect.left = (face.rect.left+1000)*width/2000;
        face.rect.top = (face.rect.top+1000)*height/2000;
        face.rect.right = (face.rect.right+1000)*width/2000;
        face.rect.bottom = (face.rect.bottom+1000)*height/2000;
        normalFace(face, width, height);
    }


    private void convertFaceToSquare(Camera.Face face){
        int width = face.rect.right-face.rect.left;
        int height = face.rect.bottom - face.rect.top;
        int size = width > height?width:height;
        int offsetH = (size - height)/2;
        int offsetW = (size - width)/2;
        int padding = 1/4*size;
        face.rect.top -= offsetH;
        face.rect.bottom += offsetH;
        face.rect.left -= offsetW;
        face.rect.right += offsetW;
        face.rect.top -= padding;
        face.rect.bottom += padding;
        face.rect.left -= padding;
        face.rect.right += padding;
        normalFace(face, 1280, 960);
    }

    private MatchWrap getBestMatchUser(float[] features){
        SqliteDb db = SqliteDb.getInstance();
        if (db != null) {
            List<SqliteDb.UserInfo> userInfos = db.getUserInfos();
            if (userInfos == null) {
                return null;
            }
            float minDistance = 100000;
            String userName = "";
            for (SqliteDb.UserInfo userInfo : userInfos) {
                float[] savedFeatures = new Gson().fromJson(userInfo.features, float[].class);
                float distance = Utils.euclideanDistance(features, savedFeatures);
                if (minDistance > distance) {
                    minDistance = distance;
                    userName = userInfo.userName;
                }
            }
            Log.e("ttttt","userName: "+userName + ",distance: "+minDistance);
            return new MatchWrap(minDistance, userName);
        }
        return null;
    }

    @Override
    public void onFaceDetection(Camera.Face[] faces, Camera camera) {
        if(faces != null && faces.length > 0 && !getFace.get() && MainActivity.mode.get() != MainActivity.MODE_IDLE){
            if(MainActivity.mode.get() == MainActivity.MODE_RECOGNIZATION){
                if(SqliteDb.getInstance() == null || SqliteDb.getInstance().getUserCount() <= 0){
                    return;
                }
            }
            int betIndex = getBestFace(faces);
            Camera.Face bestFace = faces[betIndex];
            curFace.score = bestFace.score;
            if(curFace.rect == null){
                curFace.rect = new Rect();
            }
            curFace.rect.left = bestFace.rect.left;
            curFace.rect.top = bestFace.rect.top;
            curFace.rect.right = bestFace.rect.right;
            curFace.rect.bottom = bestFace.rect.bottom;
            convertFaceIndex(curFace, 1280, 960);
            convertFaceToSquare(curFace);
            handler.sendEmptyMessage(DRAW_FACE);
            getFace.set(true);
        }else if (MainActivity.mode.get() == MainActivity.MODE_IDLE){
            handler.sendEmptyMessage(CLEAN_FACE);
        }
    }

    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) {
        if(getFace.getAndSet(false)) {
            Bitmap bitmap = fastYUV2RGB.convertYUVtoRGB(bytes, 1280, 960);
            Bitmap faceBitmap = BitmapUtils.cropFaceBitmap(bitmap, curFace);
            Bitmap rotateBitmap = BitmapUtils.rotateBitmap(faceBitmap, 270);
            Bitmap stdFaceBitmap = BitmapUtils.scaleBitmap(rotateBitmap, 224, 224);

            float[] features = faceRec.forward(stdFaceBitmap);
            if (MainActivity.mode.get() == MainActivity.MODE_REGISTER){
                if(recFeatures.size()==0){
                    FeaturesWrap wrap = new FeaturesWrap(curFace.score, features);
                    recFeatures.add(wrap);
                }else {
                    float distance = Utils.euclideanDistance(features, recFeatures.get(0).features);
                    Log.e("ttttt", "register distance: "+distance);
                    if (distance < FaceRec.DISTANCE_THRESHOLD){
                        float[] regFeatures;
                        if(recFeatures.get(0).score > curFace.score){
                            regFeatures = recFeatures.get(0).features;
                        }else {
                            regFeatures = features;
                        }
                        MatchWrap matchWrap = getBestMatchUser(regFeatures);
                        if (matchWrap != null && matchWrap.distance < FaceRec.DISTANCE_THRESHOLD){
                            if (msgProcessed.getAndSet(false)) {
                                Message msg = new Message();
                                Bundle bundle = new Bundle();
                                bundle.putString("userName", matchWrap.userName);
                                msg.what = USER_EXIST;
                                msg.setData(bundle);
                                handler.sendMessage(msg);
                            }
                        }else {
                            if (msgProcessed.getAndSet(false)) {
                                Message msg = new Message();
                                Bundle bundle = new Bundle();
                                bundle.putParcelable("image", stdFaceBitmap);
                                bundle.putString("features", new Gson().toJson(regFeatures));
                                msg.what = MainActivity.mode.get();
                                msg.setData(bundle);
                                handler.sendMessage(msg);
                            }
                        }
                    }
                    recFeatures.clear();
                }
            }else {
                MatchWrap matchWrap = getBestMatchUser(features);
                if (matchWrap != null){
                    if (matchWrap.distance < FaceRec.DISTANCE_THRESHOLD){
                        if (msgProcessed.getAndSet(false)) {
                            Message msg = new Message();
                            msg.what = MainActivity.MODE_RECOGNIZATION;
                            Bundle bundle = new Bundle();
                            bundle.putString("userName", matchWrap.userName);
                            msg.setData(bundle);
                            handler.sendMessage(msg);
                        }
                    }
                }
            }

        }
    }
}
