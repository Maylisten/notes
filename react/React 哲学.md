# React 哲学
## 页面构建流程
### 拆分组件
- 单一功能原理：一个组件理想情况下应仅做一件事情
- 小项目自上而下拆分，大项目自下而上拆分

### 构建静态版本
- 无用户交互
- 单项数据流：使用 props 传递
- 不要使用 state

### 设置 State
- 保持 [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)
	计算出你应用程序需要的绝对精简 state 表示，按需计算其它一切。举个例子，如果你正在构建一个购物列表，你可将他们在 state 中存储为数组。如果你同时想展示列表中物品数量，不需要将其另存为一个新的 state。取而代之，可以通过读取你数组的长度来实现。
- 判断是否是 state
	- 随着时间推移 **保持不变**？如此，便不是 state。
	- 通过 props **从父组件传递**？如此，便不是 state。
	- 是否可以基于已存在于组件中的 state 或者 props **进行计算**？如此，它肯定不是state、

### 寻找 state 的位置
- 通常情况下，可以直接放置 state 于它们共同的父组件。
- 也可以将 state 放置于它们父组件上层的组件。
- 如果你找不到一个有意义拥有这个 st ate 的地方，单独创建一个新的组件去管理这个 state，并将它添加到父组件上层


### 添加反向数据流
- state 在哪，改变 state 的方法就在哪
- 通过传递函数实现子组件对函数的修改

## Render
### 什么是 render
执行组件函数得到虚拟 DOM，计算与上次 render 之间的差异信息
- *组件函数是『纯函数』，所以 render 是个『纯计算』*
- *browser render 被习惯性称为 paint，来与react render区分*

### render 何时被执行 
- 首次渲染
- 当前的组件或某个父组件更新了 state

### 怎么 render 
render 就是 react 递归执行组件函数
- 初始时 
	递归执行 root.render()
- 更新时
	递归执行 state 发生变化的组件对应的组件函数，计算与上一次渲染发生变化的信息（commit 阶段使用这些信息）

### render 的作用
用新的 state 更新 UI

### 为什么要求是纯函数
 props、state、和 context 都不应该 mutate 他们，而是作为副作用 set 他们
- 可以运行在不同的环境下，比如服务器
- 可以放心的缓存
- 可以在渲染未结束时随时打断

### 平台无关性
渲染树和平台无关，在不同平台会用不同的 UI primitives 进行渲染。
### return 唯一TAG
返回 jsx 将被转义成返回对象，而一个函数只能返回一个对象，故只能返回一个 jsx tag
## Commit
### 什么是 commit
将 render 阶段生成的更新信息应用到真实 DOM 上

### commit 何时被执行 
在每次 render 之后执行 commit

### 怎么 commit
commit 只会根据 render 之间的差异选择性地改变真实 DOM，意味着*尽管组件和其子组件会重新 render（重新调用组件函数），但是真实 DOM 本身和值不一定会改变*
- 初始时 
	创建真实 DOM 树，将真实DOM 树渲染到屏幕上
- 更新时
	会根据 render 生成的更新信息，执行最小的必要操作，使真实 DOM 与虚拟 DOM 匹配


### data  和 render
每个 render 使用的总是那次 render 绑定的数据（包括 state）
但是在心理层面上，往往可以忽略这一点，将 state 视为普通数据
*更新器函数取得的是最新的 state 数据，而不是当前 render 对应的 state 数据*

### batching
React 在事件处理函数运行完毕后才会批量处理状态更新（不会等待异步）
*在一次事件中多次更新某些状态，可以使用 setNumber(n => n + 1) 更新器函数。*


## State
### 构建 state 的准则
- 组合相关状态。如果总是同时更新两个或更多状态变量，考虑将它们合并为一个单一的状态变量。
- 避免状态中的矛盾。避免多个状态相互矛盾。
- 避免冗余状态。如果在渲染过程中可以从组件的props或其现有的状态变量中计算出某些信息，不应该将那些信息放入该组件的状态中。
- 避免状态中的重复。当相同的数据在多个状态变量之间或嵌套对象中重复时，很难保持数据同步。
### 使用 Props 初始化 State
避免出现如下代码：
```jsx
function Message({ messageColor }) {  
	const [color, setColor] = useState(messageColor);
	return <></>
}
```
这种代码会使得 re-render 时 prop 时效
如果确实想让组件只有首次渲染时 prop 有效，可以在 prop 的变量加上`initial` or `default`前缀以作区分

### 重置的时机
组件或位置变化，state 值会重置
但若仅仅是 re-render，state 不会重置
*组件变化包括组件类型变化和 key 值变化*

### 组件函数内部的组件函数 state 失效
位于组件函数内部的组件函数，在父组件 re-render 时重新定义，即每次 render 时组件都发生了变化，state 无法持久存储

## Ref
### 与 State 的区别
ref 底层使用 state 实现，ref 更新时不会触发 re-render。
### 请求转发
- react 函数组件默认不支持直接接受 ref 参数，如果需要请求转发需要通过 forwardRef api 确认意图
- 低级组件可以将它们的引用转发到它们的DOM节点，高级组件通常不会暴露它们的DOM节点，以避免意外依赖于DOM结构
### useImperativeHandle
```jsx
const MyInput = forwardRef((props, ref) => {
  const realInputRef = useRef(null);
  useImperativeHandle(ref, () => ({
    focus() {
      realInputRef.current.focus();
    },
  }));
  return <input {...props} ref={realInputRef} />;
});
```

### flushSync
指示 React 在 flushSync 包裹的代码执行后立即同步更新 DOM
```jsx
flushSync(() => {
  setText('');
  setTodos([ ...todos, newTodo]);      
});
listRef.current.lastChild.scrollIntoView({
  behavior: 'smooth',
  block: 'nearest'
});
```

## Effec
### 什么是 Effect 和 side effect
side effect （副作用）指函数执行过程中，除了返回值以外，还对程序的状态或外部环境产生的额外影响。
Effect 特指 React 中渲染本身产生的而不是事件触发的 『side effect』。
### Effect 何时被执行
在 **commit 之后**执行，在 render 变动更新到真实 DOM 上后执行。
换句话说，**Effect 被执行时，DOM 已经准备好了**。




