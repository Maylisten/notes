# 使用THREEJS加载倾斜摄影
## 前置工作

无论是什么方案， 在Threejs中加载几十G的倾斜摄影模型都是不现实的，浏览器会因为超内存而崩溃

所以需要对倾斜摄影模型（osgb）进行裁剪工作

使用OSGBLab(倾斜伴侣)

链接：https://pan.baidu.com/s/1Bi5tXTi1GAyWWpewNv7k5A?pwd=uzgc 
提取码：uzgc 

![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410242026417.png)


- 消除漂浮物
- 压平
- 裁剪

**处理前效果：**

![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410242026373.png)


**处理后效果**

![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410242026291.png)


## 方案一：加载3DTiles（不推荐）

Cesium中加载倾斜摄影的常用方法是转为3dtiles后加载，自然联想到在threejs中也这样做

### 1. osbg转为3dtiles

- CesiumLab

  http://www.cesiumlab.com/cesiumlab.html

- osg2cesiumApp

  链接：https://pan.baidu.com/s/1_s9z2BarLrbbdhV9kRR8Vw?pwd=rid9 
  提取码：rid9 

- 倾斜伴侣

### 2. 加载3dtiles

使用NASA的开源库3DTilesRendererJS

https://github.com/NASA-AMMOS/3DTilesRendererJS

**注意**

3DTilesRendererJS每创建一个render只能加载一个瓦片的数据，然而倾斜摄影模型数据往往是由许许多多个瓦片组成的，所以需要递归的加载多个瓦片。

建议预先将倾斜摄影模型的索引读取并保存，前端直接读取索引文件。

- 加快加载速度
- 同时请求大量资源会导致服务器负载超过上限，从而导致加载错误

**递归创建索引文件脚本**

```js
const path = require('path');
const fs = require('fs/promises');
const url = require('url');

// 本地文件前缀，最后生成的索引文件中地址需要去掉前缀转为URL
const pathPrefix = "D:\\Project\\test-project"


async function flatModelIndex(baseJsonPath) {
    return await analyzeModelPath(baseJsonPath);
}

async function analyzeModelPath(basePath) {
    if (path.extname(basePath) !== ".json") {
        return;
    }
    const jsonData = await getJsonData(basePath);
    const children = jsonData?.root?.children;
    if (!children || children.length === 0) {
        return;
    }

    const result = { url: basePath, children: [] };
    console.log(basePath);
    for (let child of children) {
        const subPath = path.join(path.dirname(basePath), child.content.uri);
        const subItem = await analyzeModelPath(subPath)
        result.children.push(subItem);
    }

    result.children = result.children.filter(child => child)
    result.url = path2Url(result.url)
    return result;
}


async function getJsonData(basePath) {
    const strData = await fs.readFile(basePath, 'utf-8')
    const result = JSON.parse(strData.toString())
    return result;
}

function path2Url(path) {
    return path.replace(pathPrefix, "").replaceAll("\\", "/")
}


async function main() {
    const result = await flatModelIndex("D:\\Project\\test-project\\static\\tilesets_model\\tileset.json")
    await fs.writeFile("./tilesets_index.json", JSON.stringify(result))
}

main()
```

![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410242026086.png)


**前端渲染代码**

```js
    
    import * as THREE from "three";
    import { TilesRenderer } from "3d-tiles-renderer";
    const renderList = [];
     const modelGroup = new THREE.Group();
    initTiles(tileSetIndex) {
      // 初始化数据
      this._3dtilesModel = modelGroup;
      this._renderList = renderList;

      // 初始化render
     initRenderRecursion(tileSetIndex, renderList, modelGroup);

      renderList.forEach((render) => {
        window.cScene.addRenderCallback(() => {
          render.update();
        });
      });

      // modelGroup.rotation.x = (-90 * Math.PI) / 180;
      window.cScene.add(modelGroup);
    },

    initRenderRecursion(tileSetIndex, renderList, modelGroup) {
      const parentRender = generator3DTilesRender(
        this.serverBaseUrl + tileSetIndex.url
      );

      for (let child of tileSetIndex.children) {
        const subRender = this.initRenderRecursion(
          child,
          renderList,
          modelGroup
        );
        subRender.lruCache = parentRender.lruCache;
        subRender.downloadQueue = parentRender.downloadQueue;
        subRender.parseQueue = parentRender.parseQueue;
      }
      renderList.push(parentRender);
      modelGroup.add(parentRender.group);
      return parentRender;
    },

    generator3DTilesRender(url) {
      const tilesRenderer = new TilesRenderer(url);

      tilesRenderer.manager.addHandler(
        /\.gltf$/,
        window.cScene.dracoGltfLoader
      );

      tilesRenderer.manager.addHandler(/\.drc$/, window.cScene.dracoGltfLoader);

      tilesRenderer.setCamera(window.cScene.renderCamera);
      tilesRenderer.setResolutionFromRenderer(
        window.cScene.renderCamera,
        window.cScene.renderer
      );
      return tilesRenderer;
    },

    removeModel() {
     renderList?.forEach((render) => {
        render.dispose();
      });

      window.cScene.removeObject(this._3dtilesModel);
      this._renderList = [];
      this._3dtilesModel = null;
    }
```



## 方案二：加载GLTF（推荐）

### 1. 转为OBJ

使用倾斜伴侣

将每个瓦片转为一个obj

![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410242026558.png)


### 2. obj转为gltf

使用obj2gltf将每个obj转为gltf

https://github.com/CesiumGS/obj2gltf

### 3. 合并多个gltf

使用threejs官方提供的edit编辑器即可

导入多个gltf后添加一个group包裹，然后调整模型位置为（0,0,0,）

![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410242027723.png)


### 3. 加载gltf

不过多赘述

