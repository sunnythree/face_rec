package com.example.myapplication;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ThreadPoolFactory {
    private static final String TAG = "faceRec.ThreadPoolFactory";
    private ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(2, 4, 3, TimeUnit.SECONDS, new ArrayBlockingQueue<Runnable>(100));
    private static ThreadPoolFactory instance;
    public static ThreadPoolFactory getInstance(){
        if (instance == null){
            synchronized (ThreadPoolFactory.class){
                if (instance == null){
                    instance = new ThreadPoolFactory();
                }
            }
        }
        return instance;
    }
    public void execute(Runnable runnable){
        threadPoolExecutor.execute(runnable);
    }

}
