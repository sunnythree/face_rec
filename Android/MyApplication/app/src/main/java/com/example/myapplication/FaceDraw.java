package com.example.myapplication;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.view.View;

public class FaceDraw extends View {
    private Paint mPaint;
    private Rect drawRect;
    private boolean draw = false;

    public FaceDraw(Context context) {
        super(context);
        // TODO Auto-generated method stub
        mPaint = new Paint();
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeWidth(5);
        mPaint.setColor(Color.BLUE);
        drawRect = new Rect();
    }

    private String printRect(Rect rect){
        StringBuilder sb = new StringBuilder();
        sb.append(rect.left).append(",");
        sb.append(rect.top).append(",");
        sb.append(rect.right).append(",");
        sb.append(rect.bottom).append(",");
        return sb.toString();
    }

    private void converIndexToDraw(Rect rect){
        float scale = getHeight()/(float)1280;
        int h = rect.bottom - rect.top;

        drawRect.left = 960 - rect.bottom;
        drawRect.right = 960 - rect.bottom + h;
        drawRect.top = 1280 - rect.left;
        drawRect.bottom = 1280- rect.right;

        drawRect.left = (int) (drawRect.left*scale);
        drawRect.right = (int) (drawRect.right*scale);
        drawRect.top = (int) (drawRect.top*scale);
        drawRect.bottom = (int) (drawRect.bottom*scale);
        //Log.d("sssss",printRect(rect)+","+printRect(drawRect));
    }
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if(draw) {
            if (CameraCallBack.curFace != null && CameraCallBack.curFace.rect != null) {
                converIndexToDraw(CameraCallBack.curFace.rect);
                canvas.drawRect(drawRect, mPaint);
            }
        }
        canvas.drawRect(new Rect(getLeft(),getTop(),getRight(), getBottom()), mPaint);
    }

    public void draw(){
        draw = true;
        invalidate();
    }

    public void clear(){
        draw = false;
        invalidate();
    }
}
