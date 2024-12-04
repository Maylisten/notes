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

### render 触发新的 render
当在渲染过程中更新组件时，React会丢弃返回的JSX并立即重新尝试渲染
*所以尽量在 组件函数中 set 而不是在 useEffect 中*


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

## Effect
### 什么是 Effect 和 side effect
side effect （副作用）指函数执行过程中，除了返回值以外，还对程序的状态或外部环境产生的额外影响。
Effect 特指 React 中渲染本身产生的副作用，而不是交互产生的副作用。
### Effect 何时被执行
在 **commit 之后**执行，在 render 变动更新到真实 DOM 上后执行。
换句话说，**Effect 被执行时，DOM 已经准备好了**。

### Effect 的作用
只有在因为组件渲染展示而产生副作用时，使用 Effect，其他使用 event handler、

### useEffect 不需要的依赖
- ref 对象
	ref 对象在 React 中是视为恒定不变的
- setState
	setState 方法在 React 中是视为恒定不变的

### useEffect 常见使用场景
- 控制非 react 的小部件
- 订阅事件
- 触发动画
- 远程请求数据
- 埋点

### useEffect 不建议的使用场景
**不应该将 useEffect 当做 vue 中的 watch 使用**
如果必须使用 useEffect，尝试将他封装到自定义 hook 中。越少的使用原始的 `useEffect` ，代码越好管理d

| **场景**                           | **错误原因**                        | **正确方案**                                                          |
| -------------------------------- | ------------------------------- | ----------------------------------------------------------------- |
| 一次性操作，如获取用户设备信息                  | 应用只执行一次，不应该放在组件中                | 放在组件外部，如何入口文件顶部                                                   |
| 购买商品                             | effect 执行不可控，可能执行多次             | 在 event handler中购买商品                                              |
| 计算属性                             | 会触发 re-render，影响性能              | 直接在组件函数中计算，re-render 时会重新计算                                       |
| 计算耗时属性                           | 会触发 re-render，影响性能              | 使用 useMemo 缓存计算结果                                                 |
| 组件参数变化，重置全部 state                | 会触发 re-render，影响性能              | 给组件添加 key，key 变化自动重置 state                                        |
| 组件参数变化，调整或重置部分 state             | 会触发 re-render，影响性能              | 存储上一次 render 的参数，如果变化则重新 setState（render 中 setState 会立即重新 render） |
| 通过监听state，共享 event handler 之间的逻辑 | 代码扩展性差，当 state 在新的场合中改变时，可能出现问题 | 提取 event handler 的公共逻辑函数，仍在 event handler 中触发                     |
| 避免链式计算                           | 串行计算，多次 re-render，影响性能          | 将计算逻辑提取到 event handler 中                                          |


### cleanup function
当 effect 被再次执行或者组件被卸载时，cleanup function 会被执行


### useEffect 与函数闭包
useEffect 中如果使用了函数组件 body 中声明的函数，务必注意需要将函数添加到依赖项中
函数组件 body 中声明的函数也是 **reactive** 的，使用的数据是声明时对应的 render 的数据 snapshot，也就是产生了闭包！当组件 re-render，函数重新被定义，就需要重新触发 Effect。
**最好的做法是将需要使用的函数直接定义在 useEffect 中**

### Effect 中使用 setState
在 useEffect 中使用 setState，需要通过传入函数的方式避免依赖更新的 state，例如：
```jsx
function ChatRoom({ roomId }) {  
	const [messages, setMessages] = useState([]);  
	useEffect(() => {  
		const connection = createConnection();
		connection.connect();
		connection.on('message', (receivedMessage) => {
		setMessages(msgs => [...msgs, receivedMessage]);  
	});  
	return () => connection.disconnect();
	}, [roomId]); // ✅ All dependencies declared  
	// ...
}
```

### Effect Event
非响应式，但是使用的 state 和 prop 永远是最新一次 render 的绑定的，用于分离Effect 中**响应式的逻辑和非响应式的逻辑**
*与之对应的，在 render 中定义的普通函数，将再下一次  render  时重新定义。而其使用的 state 或 prop，是当前  render 所绑定的*

例如，需要统计用户打开页面时购物车商品的数量，是有 url 变化时需要重新统计，而 numberOfItems 变化时不需要：
```jsx
function Page({ url }) {  
	const { items } = useContext(ShoppingCartContext);  
	const numberOfItems = items.length;  
	const onVisit = useEffectEvent(visitedUrl => {  
	logVisit(visitedUrl, numberOfItems);  
	});  
	useEffect(() => {  
		onVisit(url);  
	}, [url]); // ✅ All dependencies declared  
	// ...  
}
```

## 异步请求数据
###  race condition
多个任务异步访问资源，无法确认最终资源对应哪一个访问。
解决方法：
- useEffect Clean-up Function with **boolean flag**
```jsx
useEffect(() => {
  let active = true;
  const fetchData = async () => {
    setTimeout(async () => {
      const response = await fetch(`https://swapi.dev/api/people/${props.id}/`);
      const newData = await response.json();
      if (active) {
        setFetchedId(props.id);
        setData(newData);
      }
    }, Math.round(Math.random() * 12000));
  };
  fetchData();
  return () => {
    active = false;
  };

}, [props.id]);
```
- useEffect Clean-up Function with **AbortController**
```jsx
useEffect(() => {
  const abortController = new AbortController();
  const fetchData = async () => {
    setTimeout(async () => {
      try {
        const response = await fetch(`https://swapi.dev/api/people/${id}/`, {
          signal: abortController.signal,
        });
        const newData = await response.json();
        setFetchedId(id);
        setData(newData);
      } catch (error) {
        if (error.name === 'AbortError') {
          // Aborting a fetch throws an error

          // So we can't update state afterwards
        }
        // Handle other request errors here
      }
    }, Math.round(Math.random() * 12000));
  };
  fetchData();
  return () => {
    abortController.abort();
  };

}, [id]);	
```

### 直接 fetch 数据的缺陷
- 不能用于服务端渲染
- 容易产生 `network waterfalls`(大量的串行请求)
- 没有预加载和缓存
- 完善的请求写起来非常繁琐

### 请求数据的正确姿势
- 若使用框架，则使用框架自带的数据请求方案
- 若未使用框架，推荐使用[React Query](https://tanstack.com/query/latest), [useSWR](https://swr.vercel.app/), [React Router 6.4+.](https://beta.reactrouter.com/en/main/start/overview)(将数据加载提前到路由)


## Hooks

### useEffectEvent VS useCallback 
| useEffectEvent                   | useCallback                |
| -------------------------------- | -------------------------- |
| 返回结果永远不变                         | 返回结果根据依赖项变化而变化             |
| 永远使用最新的 state 和 prop             | 使用当前 render 的 state 和 prop |
| 不可向下传递                           | 可以向下传递                     |
| 用于分离 useEffect 中 no-reactive 的逻辑 | 用于缓存函数                     |

### Custom  Hooks 构建规范
- 以 use 开头命名
- 必须为纯函数
- 必须（或将来）使用 React 内置 Hook 或别的自定义 Hook
- useEffect 应该总是在自定义 Hook 中出现
- 将接收到的 event handler  使用 useEffectEvent 包裹
- 避免使用 useMount 等自定义的声明周期类自定义 Hook，这样会导致 linter 无法帮助分析 useEffect code 中的依赖
- 从具体使用的场景出发去封装自定义 hook，而不是站在通用的角度出发
