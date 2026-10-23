/* A cancellable client for fresh computation in the visitor's own browser. */
(()=>{
  'use strict';
  let worker=null, sequence=0, generation=0, initialized=null, info=null;
  const pending=new Map();
  function failAll(error){for(const job of pending.values())job.reject(error);pending.clear();}
  function createWorker(){
    if(worker)return worker;
    if(!window.Worker||!window.WebAssembly)throw new Error('此浏览器不支持模型计算所需的 Web Worker / WebAssembly，请使用较新的系统浏览器。');
    const currentGeneration=generation;
    worker=new Worker(new URL('compute-worker.js',document.baseURI),{type:'module'});
    worker.onmessage=event=>{
      if(currentGeneration!==generation)return;
      const data=event.data||{},job=pending.get(data.requestId);
      if(!job)return;
      if(data.type==='progress'){job.onProgress?.(data);return;}
      pending.delete(data.requestId);
      if(data.type==='error'){const error=new Error(data.message||'浏览器计算失败');error.detail=data.detail;job.reject(error);}
      else if(data.type==='ready'){info=data.info;job.resolve(info);}
      else if(data.type==='result')job.resolve(data.result);
      else job.reject(new Error('计算模块返回了未知消息'));
    };
    worker.onerror=event=>{
      event.preventDefault();
      if(currentGeneration!==generation)return;
      const error=new Error(event.message||'计算模块未能启动，请检查网络后重试。');
      failAll(error);worker?.terminate();worker=null;initialized=null;info=null;generation++;
    };
    return worker;
  }
  function request(type,data,onProgress){
    return new Promise((resolve,reject)=>{
      try{
        const target=createWorker(),requestId=++sequence;
        pending.set(requestId,{resolve,reject,onProgress});
        target.postMessage({type,requestId,...data});
      }catch(error){reject(error);}
    });
  }
  function init(onProgress){
    if(!initialized)initialized=request('init',{},onProgress).catch(error=>{worker?.terminate();worker=null;info=null;initialized=null;generation++;failAll(error);throw error;});
    return initialized;
  }
  async function run(config,{compare=false,onProgress}={}){
    await init(onProgress);
    return request('run',{config,compare},onProgress);
  }
  function cancel(){
    generation++;worker?.terminate();worker=null;initialized=null;info=null;
    failAll(new DOMException('已取消本次计算，保留上一组结果。','AbortError'));
  }
  window.DeliveryCompute={init,run,cancel,getInfo:()=>info};
})();
