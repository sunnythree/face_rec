package com.example.myapplication;

import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Base64;
import android.util.Log;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;

public class SqliteDb extends SQLiteOpenHelper {
    private SQLiteDatabase db =null;
    private static final String TAG = "faceRec.SqliteDb";
    private static final String DATABASE_NAME = "user";
    private static SqliteDb mInstance=null;
    private List<UserInfo> userInfos;

    public static class UserInfo{
        String userName;
        String features;
    }

    public static SqliteDb getInstance(){
        return mInstance;
    }

    public static SqliteDb getInstance(Context context){
        if (mInstance==null){
            synchronized (SqliteDb.class){
                if (mInstance==null){
                    mInstance = new SqliteDb(context, 1);
                }
            }
        }
        return mInstance;
    }

    public SqliteDb(@Nullable Context context, int version) {
        super(context, DATABASE_NAME, null, version);
    }

    @Override
    public void onCreate(SQLiteDatabase sqLiteDatabase) {
    }

    @Override
    public void onUpgrade(SQLiteDatabase sqLiteDatabase, int i, int i1) {

    }


    public void openDb(boolean writeable){
        if(writeable) {
            db = getWritableDatabase();
        }else {
            db = getReadableDatabase();
        }
        createUserLogInfoTable();
        if(userInfos == null) {
            userInfos = queryUserList();
        }
    }

    public boolean createUserLogInfoTable(){
        if (db == null){
            Log.e(TAG, "database not open error");
            return false;
        }
        try{
            db.execSQL("create table if not exists UserInfo (_id INTEGER PRIMARY KEY AUTOINCREMENT,userName VARCHAR,features VARCHAR)");
        } catch(SQLException e) {
            return false;
        }
        return true;
    }

    public boolean insertOneUser(String userName, String features) {
        if (db == null){
            Log.e(TAG, "database not open error");
            return false;
        }
        try{
            db.execSQL("insert into UserInfo values(NULL,?,?)",new Object[]{userName, features});
        } catch(SQLException e) {
            return false;
        }
        updateUserInfos();
        return true;
    }

    public boolean deleteTopUser(String userName) {
        if (db == null){
            Log.e(TAG, "database not open error");
            return false;
        }
        try {
            db.execSQL("delete from UserInfo where userName=?", new Object[]{userName});
        } catch(SQLException e) {
            Log.d(TAG,"deleteTopLogInfo fail");
            return false;
        }
        updateUserInfos();
        return true;
    }

    public boolean clear(){
        if (db == null){
            Log.e(TAG, "database not open error");
            return false;
        }
        try {
            db.execSQL("delete from UserInfo");
        } catch(SQLException e) {
            Log.d(TAG,"deleteTopLogInfo fail");
            return false;
        }
        updateUserInfos();
        return true;
    }

    public boolean update(String userName, String features) {
        if (db == null){
            Log.e(TAG, "database not open error");
            return false;
        }
        try {
            db.execSQL("update UserInfo set userName=? where features=?", new Object[]{userName, features});
        } catch(SQLException e) {
            Log.d(TAG,"deleteTopLogInfo fail");
            return false;
        }
        updateUserInfos();
        return true;
    }

    public List<UserInfo> queryUserList() {
        if (db == null){
            Log.e(TAG, "database not open error");
            return null;
        }

        List<UserInfo> lstData;
        lstData = new ArrayList<UserInfo>();
        Cursor c = null;
        c = db.rawQuery("SELECT * FROM UserInfo", null);
        while(c.moveToNext()) {
            UserInfo  tmpData= new UserInfo();
            tmpData.userName = c.getString(c.getColumnIndex("userName"));
            tmpData.features = c.getString(c.getColumnIndex("features"));
            lstData.add(tmpData);
            Log.d(TAG,tmpData.toString()+"id: "+c.getInt(c.getColumnIndex("_id")));
        }
        c.close();
        return lstData;
    }

    public int getUserCount(){
        if (db == null){
            Log.e(TAG, "database not open error");
            return -1;
        }

        Cursor c = null;
        c = db.rawQuery("SELECT * FROM UserInfo", null);
        int count = c.getCount();
        Log.d(TAG,"count: "+count);
        c.close();
        return count;
    }

    public void updateUserInfos(){
        userInfos = queryUserList();
    }

    public List<UserInfo> getUserInfos(){
        return userInfos;
    }
}
