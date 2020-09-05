package com.example.myapplication;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.Image;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.torchvision.TensorImageUtils;

import java.util.concurrent.atomic.AtomicInteger;

import static androidx.core.content.PermissionChecker.PERMISSION_GRANTED;


public class MainActivity extends AppCompatActivity implements  SurfaceHolder.Callback{
    private SurfaceView surfaceView;
    private Button buttonReg;
    private Button buttonRec;
    private Button buttonCleanAll;
    public static final int MODE_IDLE = 0;
    public static final int MODE_REGISTER = 1;
    public static final int MODE_RECOGNIZATION = 2;
    public static AtomicInteger mode = new AtomicInteger(MODE_IDLE);
    public static Context context = null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        context = this;
        surfaceView = findViewById(R.id.camera_sf);
        surfaceView.getHolder().addCallback(MainActivity.this);
        buttonRec = findViewById(R.id.bt_recognization);
        buttonReg = findViewById(R.id.bt_register);
        buttonCleanAll = findViewById(R.id.bt_clean_all);
        buttonRec.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mode.set(MODE_RECOGNIZATION);
                Toast.makeText(MainActivity.this, "recognization mode", Toast.LENGTH_SHORT).show();
            }
        });
        buttonReg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mode.set(MODE_REGISTER);
                Toast.makeText(MainActivity.this, "register mode", Toast.LENGTH_SHORT).show();
            }
        });
        buttonCleanAll.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(SqliteDb.getInstance() != null){
                    SqliteDb.getInstance().clear();
                    Toast.makeText(MainActivity.this, "all user have been deleted!!!", Toast.LENGTH_SHORT).show();
                }
            }
        });
        ThreadPoolFactory.getInstance().execute(new Runnable() {
            @Override
            public void run() {
                SqliteDb.getInstance(getApplicationContext()).openDb(true);
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PERMISSION_GRANTED) {
            MCamera.getInstance().startPreview(surfaceView.getHolder());
        }  else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                    11);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (SqliteDb.getInstance() != null) {
            SqliteDb.getInstance().close();
        }
        context = null;
    }

    @Override
    public void surfaceCreated(@NonNull SurfaceHolder surfaceHolder) {
        Log.d("ttt","surfaceCreated");
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PERMISSION_GRANTED) {
            Log.d("ttt","startPreview");
            MCamera.getInstance().releaseCamera();
            MCamera.getInstance().startPreview(surfaceHolder);
        }
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder surfaceHolder, int i, int i1, int i2) {
        Log.d("ttt","surfaceChanged");

    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder surfaceHolder) {
        Log.d("ttt","surfaceDestroyed");
        MCamera.getInstance().releaseCamera();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == 11){
            if(grantResults.length > 0 && grantResults[0]==PERMISSION_GRANTED){
                if(permissions[0].equals(Manifest.permission.CAMERA)){
                    MCamera.getInstance().startPreview(surfaceView.getHolder());
                }
            }
        }
    }
}