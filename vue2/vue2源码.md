# vue2源码
## Wather

实际上，Watcher对象的作用有两点：

- 配合dep实现数据变化的监听
- 通过类似防抖的机制实现批量的回调执行

其中的属性和方法含义：

- value：监听数据的值
- id：用于批量处理时去重
- options.user：标识是内部创建的wather还是用户创建的watcher
- cb： 数据更新时会自动调用的回调函数
- get：获取新数据，非lazy时，wathcer对象在构建时就会调用一次get方法，并设置value为get的返回值
- deps： 记录相关的dep对象
- update：调用此方法会触发watcher的更新逻辑，即get方法获取新数据更新value，调用cb回调函数

总而言之，watcher的作用就是，当数据发生改变时，会被（dep）调用update函数，从而执行get方法获取新的value，调用cb回调

## dep

data中每一个属性都会有一个与之对应的dep对象

* depend：将当前的wathcer记录下来
* notify：触发相关watcher的更新

## 数据劫持

data中每一个属性都会有一个与之对应的dep对象

- 触发属性的get时，将dep对象和Dep.target(当前的watcher对象)进行一个双向的记录（多对多）(dep.depend())

- 触发属性的set时，会调用dep对象记录的所有watcher的update方法（dep.notify()）

通过以上两个步骤，就实现了监听数据变化

## 自动更新

将用于更新组件视图的updateComponent方法包装成一个watcher对象

此时watcher对象获取新数据的方法（get）会被设置为updateComponent方法。也就是说当数据发生变化时，会触发updateCompoent方法，从而实现组件视图的更新

## watch

watch也是通过创建watcher对象实现的

watcher的get方法设置获取最新的数据。 注意：当监听嵌套数据时，如“student.address.name”，get方法会循环调用获取到最终的属性值，即student、adress、name属性所对应的dep对象都会和当前的watcher对象建立联系

wathcer的cb方法设置为监听时的处理逻辑

当数据发生变化时，update方法会被调用，通过get方法更新value，并把新值和旧值传递给cb函数

## patch

1. 首次patch，oldVnode传的是真实dom节点，特殊处理，直接创建真实dom挂载
2. 后续更新时，传入旧的vnode（vm._vnode）和新的vnode

2. 对于父节点

   - 标签不一致直接创建真实dom替换
   - 都为文本节点，直接替换文本节点内容
   - 标签一致且都为元素节点，更新属性之后，继续updateChildren

3. updateChildren使用diff算法，步骤如下：

   1. 交叉比对
      - 每层循环都遵循首首，尾尾，首尾，尾首的顺序匹配
      - 匹配标准是key和tag都相同
      - 尽可能的循环到四个指针都匹配不上，才跳出循环
      - 匹配上的递归执行patch算法

   2. 暴力比对

      - 循环没没交叉比对上的新子虚拟节点

      - 会先对旧的虚拟子节点创建key—index的索引map
      - 如果旧节点中有匹配的节点，则重复利用（移动）真实dom
      - 没有匹配的节点，则根据新的节点创建真实dom
      - 将真实dom插入到新子节点索引对应的位置
      - 如果服用真实dom，则需要在移动真实dom后，将旧虚拟节点列表中原位置设置为null，防止高度塌陷

   3. 删除多余真实节点

      - 循环没交叉比对上的旧子虚拟节点，挨个删除

​		

## computed

- computed的实现还是基于watcher和dep，会为每一个计算属性创建一个wathcher对象

- 此时的watcher的lazy为true，同时会有一个dirty变量记录此时的值是否为新值

- 在计算属性获取时，会看看对应的wathcer的dirty

  - 如果dirty为false，则直接返回watcher的value；

  - 如果dirty为true，则调用watcher的evaluate方法，该方法会重新计算wathcer.value的值，并将dirty的值设置为false

- 由于计算计算属性的值的时候，会获取依赖的属性的值，所以自然计算属性的watcher，会被依赖的属性的dep收集到
- **如果依赖的属性没有在视图中，那么该属性变化就不会重新渲染模板。但如果计算属性依赖了该属性，计算属性就会在每次计算值的时候，主动将依赖的属性的dep和渲染模板的watcher建立双向联系。如此当计算属性依赖的任意属性发生变化时，模板都会重新渲染，计算属性的值也就会重新获取计算。**



