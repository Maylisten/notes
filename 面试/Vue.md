# Vue
## $router 与 $route 的区别

- $route 对象表示当前的路由信息，包含了当前 URL 解析得到的信息
    - $route.path：字符串，对应当前路由的路径，总是解析为绝对路径，如 "/foo/bar"。
    - $route.params： 一个 key/value 对象，包含了 动态片段 和 全匹配片段，如果没有路由参数，就是一个空对象。
    - $route.query：一个 key/value 对象，表示 URL 查询参数。例如对于路径 /foo?user=1，则有 $route.query.user == 1
      ，如果没有查询参数，则是个空对象。

- $router 对象是全局路由的实例
    - push：向 history 栈添加一个新的记录
    - replace：替换当前的页面，不会向 history 栈添加一个新的记录
    - go：页面路由跳转前进或者后退

## hash 模式和 history 模式的实现原理

- hash 模式通过监听 hashchange 事件可以知道 hash 发生了哪些变化，然后根据 hash 变化来实现更新页面部分内容的操作。

- history pushState 和 replaceState，这两个 API 可以在改变 URL 但是不会发送请求，并监听 popState 的事件

两种模式的区别：

- 首先是在 URL 的展示上，hash 模式有“#”，history 模式没有

- 刷新页面时，hash 模式可以正常加载到 hash 值对应的页面，而 history 没有处理的话，会返回 404，一般需要后端将所有页面都配置重定向到首页路由

- hash模式兼容性更好

## vueRouter 有哪几种导航守卫

- 全局前置/钩子：beforeEach、beforeResolve、afterEach

- 路由独享的守卫：beforeEnter

- 组件内的守卫：beforeRouteEnter、beforeRouteUpdate、beforeRouteLeave

一次完整的导航解析流程如下：

1. 导航被触发。
2. 在失活的组件里调用离开守卫。
3. 调用全局的 beforeEach 守卫。
4. 在重用的组件里调用 beforeRouteUpdate 守卫（2.2+）。
5. 在路由配置里调用 beforeEnter。
6. 解析异步路由组件。
7. 在被激活的组件里调用 beforeRouteEnter。
8. 调用全局的 beforeResolve 守卫（2.5+）。
9. 导航被确认。
10. 调用全局的 afterEach 钩子。
11. 触发 DOM 更新。
12. 用创建好的实例调用 beforeRouteEnter 守卫中传给 next 的回调函数

## v-if 与 v-show 的区别

- v-if 可以控制对应的虚拟dom元素创不创建，真实dom元素生不生成，有更高的切换消耗

- v-show 是通过设置真实 DOM 元素的 display 样式属性控制显隐

## scoped 的实现原理

vue 中的 scoped 属性的效果主要通过 PostCSS 转译实现的。PostCSS 给一个组件中的所有 DOM 添加了一个独一无二的动态属性，然后，给CSS
选择器额外添加一个对应的属性选择器来选择该组件中 DOM，这种做法使得样式只作用于含有该属性的 DOM，即组件内部 DOM。

## keep-alive 的作用和实现原理

keep-alive 是一个内置组件，主要用于缓存不活动的组件实例，而不是将它们销毁

- 作用
    - 性能优化：缓存不活动的组件实例，避免重复渲染，提高应用性能
    - 保持状态：在组件切换时保留组件的状态和 DOM
    - 增强用户体验：在组件切换时保持滚动位置、表单输入等状态，提升用户体验
- 使用方式
    - activated：当组件被 keep-alive 缓存并重新激活时调用。
    - deactivated：当组件被 keep-alive 缓存但不再显示时调用。
    - include 和 exclude：通过这些属性，可以指定哪些组件应该被缓存，哪些不应该被缓存。
- 实现原理（render函数逻辑               
  keep-alive组件实例中会维持一个cache和一个keys数组
    1. 获取keep-alive默认插槽中第一个子VNode
    2. 根据设定的黑白名单（如果有）进行条件匹配，决定是否缓存。不匹配，直接返回VNode，否则执行第三步；
    3. 在缓存对象中查找是否已缓存过该VNode。如果存在，直接取出缓存值并放到最后（更新key的位置是实现LRU置换策略的关键)
    4. 在this.cache对象中存储该组件实例并保存key值，之后检查缓存的实例数量是否超过max设置值，超过则根据LRU置换策略删除最近最久未使用的实例（即是下标为0的那个key）
    5. 最后并且很重要，将该组件实例的keepAlive属性值设置为true。（当`vnode.componentInstance`和`keepAlive`
       同时为true时，不再进入$mount过程，那mounted之前的所有钩子函数（beforeCreate、created、mounted）都不再执行，并且会执行activated和deactivated

## Vue 组件的 data 为什么必须是函数

组件中的 data 写成一个函数，数据以函数返回值形式定义。这样每复用一次组件，就会返回一份新的 data

## Vue Computed 实现原理

1. 组件挂载时，在beforeCrated之后，created之前，会对组件的状态进行处理，其中就包括computed中的数据
2. 在处理时，会为每一个computed属性创造一个lazy的watcher对象，初始dirty属性为ture（且get函数不会立即执行），有个value属性保存当前值。
3. vue会将计算属性代理到实例上。
4. 当render函数被调用，获取computed属性时，会检查watcher的dirty，如果为false，则直接返回value；否则调用computed设置的get方法
5. 由于get时会利用到prop或data中的数据，这个watcher就会被对应数据的dep收集到，当对应数据发生改变时，派发更新，watcher中dirty属性变为true。
6. 数据变了，组件render肯定会重新调用，然后又会重新获取computed计算结果
7. 至于 set 方法就很简单，当设置属性时直接调用 set 函数就可以，如果修改了依赖的数据，又会响应式更新。

## Vue watch 实现原理

1. 每个watch创建一个watcher对象
2. 会将字符串key代表的属性，层层调用直到调用到对应属性，过程中的dep都会收集到这个watcher对象作为依赖
3. 只要属性变化，watcher自动触发逻辑执行

## vue 如何快速定位那个组件出现性能问题的

⽤ timeline ⼯具。 通过 timeline 来定位出哪个函数的问题，从⽽能判断哪个组件出了问题。

## proxy相比于definedProperty的优势是什么

Vue3.x 改用 Proxy 替代 Object.defineProperty

原因在于 Object.defineProperty 本身存在的一些问题：

- Object.defineProperty 只能劫持对象属性的 getter 和 setter 方法。
- Object.definedProperty 不支持数组

而相比 Object.defineProperty，Proxy 的优点在于：

- Proxy 是直接代理劫持整个对象。
- Proxy 可以直接监听对象和数组的变化，并且有多达 13 种拦截方法。

目前，Object.definedProperty 唯一比 Proxy 好的一点就是兼容性，不过 Proxy 新标准也受到浏览器厂商重点持续的性能优化当中。

## 异步请求一般放在哪个生命周期中

created中

- 尽早获取
- ssr没有beforeMount、mounted钩子，created可以做到统一

## 50. 说一说自定义指令有哪些生命周期？

bind -> beforeMount

inserted -> mounted

无 -> beforeUpdate（Vue 3 新增）

update -> updated

无 -> beforeUnmount（Vue 3 新增）

unbind -> unmounted

## vue3.x 对比 vue2.x 变化

1. 源码组织方式变化：使用 TS 重写
2. 支持 Composition API：基于函数的API，更加灵活组织组件逻辑（vue2用的是options api）
3. 响应式系统提升：Vue3中响应式数据原理改成proxy，可监听动态新增删除属性，以及数组变化
4. 编译优化：vue2通过标记静态根节点优化diff，Vue3 标记和提升所有静态根节点，diff的时候只需要对比动态节点内容
5. 打包体积优化：移除了一些不常用的api（inline-template、filter）
6. 生命周期的变化：使用setup代替了之前的beforeCreate和created
7. Vue3 的 template 模板支持多个根标签
8. Vuex状态管理：创建实例的方式改变,Vue2为new Store , Vue3为createStore
9. Route 获取页面实例与路由信息：vue2通过this获取router实例，vue3通过使用
   userRoute和userRouter方法获取当前组件实例
10. Props 的使用变化：vue2 通过 this 获取 props 里面的内容，vue3 直接通过 props
11. 自定义事件需要提前声明一下
12. 生命周期更名

- beforeCreate -> setup（Vue 3 引入的 Composition API，将状态逻辑前置到 setup 中）
- created -> setup（同上，setup 包含了 beforeCreate 和 created 阶段的逻辑）
- beforeMount -> onBeforeMount
- mounted -> onMounted
- beforeUpdate -> onBeforeUpdate
- updated -> onUpdated
- beforeDestroy -> onBeforeUnmount
- destroyed -> onUnmounted
- activated -> onActivated
- deactivated -> onDeactivated
- errorCaptured -> onErrorCaptured



## 组件中写 name 作用

1. 组件递归

2. keep-alive黑白名单

3. 调试和编辑器识别

   

## vue-router原理

1. 明确VueRouter的使用方法（写法），会先`use（VueRouter）`，还会创建VueRouter的实例`new VueRouter({mode,routes})`，并且`new Vue({router})`
2. 根据使用方法得知，VueRouter是一个类，有个静态方法install，创建Vue实例之前，会先运行这个install方法。VueRouter传递mode和routes（路由配置）参数创建实例，会作为参数传到`new Vue（{router}）`中

3. VueRouter创建的实例时，会运行init方法，init方法的作用就是监听路由的变化，**hash就监听`hashChange`事件，history就监听`popState`对象，会在实例中使用current属性保存当前路由**
4. 在创建Vue实例的时候，会将router作为参数传入，目的是让intall方法调用时拿到
5. install方法会先混入一个全局的`beforeCreated`生命周期钩子函数，将router对象挂载到Vue实例和Vue组件实例的`$router`上，具体说就是Vue实例就通过$options中拿到router对象（`new Vue（{router}）`传入的参数），vue组件实例就拿父节点的router。
6. install方法会注册全局的组件，`router-link`和`rouer-view`
   - router-link原理很简单，就是渲染一个a标签，通过参数to的值，进行页面跳转
   - router-view实现，根据步骤5可知通过`this.$router`能拿到VueRouter实例对象，通过$router的current和routes查找匹配的vue组件，然后`h`函数渲染。而current又是响应式的，发生变化之后，render函数重新调用，组件也就对应的发生变化

总体流程

**url变化->router对象中的响应式属性current变化->使得router-view组件重新渲染**
