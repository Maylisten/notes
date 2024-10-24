# React
## 什么是React

在过去，前端都是使用Jquery直接操作dom，但是随着单页面应用的流行代码量变多，传统方式就显得太过繁琐。

这时就出现了 Vue、React这样的前端框架。

实际上， Vue、React 这些前端框架的核心是 “构建UI” 的库，主要是提供两个功能：

基于状态的声明式渲染（MVVM）
基于组件的开发
随着前端SPA（单页面应用）的发展，需要路由解决方案，比如vue-router，react-router

随着组件的数量变多，就需要状态管理，比如redux，vuex

把这些解决方案（包括库本身），结合在一起可以称作框架，也可以叫做技术栈。

- 基于状态的**声明式**渲染
- 支持组件化开发（复用）
- 前端路由方案（随着单页应用诞生）
- 状态管理方案

## Context 上下文的使用方式

1. createContext得到context对象
2. context.Provider组件包裹子组件，value传入上下文共享的数据
3. 子组件useContext传入context作为参数得到上下文数据

Parent.tsx

```tsx
import {createContext, useState} from "react";
import Son from "./Son";

type Person = {
    name: string,
    age: number
}

type ContextType = {
    person: Person,
    setPerson: (value: Person) => void,
}

export const MyContext = createContext<ContextType | undefined>(undefined);


function Parent() {
    const [person, setPerson] = useState<Person>({name: "Mayer", age: 1})

    return (
        <MyContext.Provider value={{person, setPerson}}>
            <Son/>
        </MyContext.Provider>
    )
}


export default App
```

Son.tsx

```tsx
import {useContext} from "react";
import {MyContext} from "./Parent";

function Son() {
    const {person, setPerson} = useContext(MyContext)!;

    return (
        <div>
            name: {person.name}
            <br/>
            age: {person.age}
            <br/>
            <button type="button" onClick={() => {
                setPerson({...person, age: person.age + 1})
            }}>年龄加1
            </button>
        </div>
    );
};

export default HelloWord;
```

## Redux使用方法

1. 从@reduxjs/toolkit包中引入createSlice，传入初始状态和reducer函数创建slice对象
2. 从slice对象中导出actions函数和reducer
3. 从@reduxjs/toolkit中导入configureStore方法，传入若干slice对象作为参数，导出得到的store对象
4. 从react-redux中导出Provider组件，包裹住根组件，传入store作为参数
5.

子组件中使用useSelector钩子从store中取值使用，使用useDispatch获取dispatch对象，通过actions函数获得对应的action对象，dispatch(
action)触发reducer函数

store/catSlice.ts

```ts
import {createSlice} from "@reduxjs/toolkit";

const catSlice = createSlice({
    name: "cat",
    initialState: {
        age: 1
    },
    reducers: {
        grow: (state) => {
            state.age += 1;
        }
    }
})

export const {grow} = catSlice.actions;

export default catSlice.reducer;
```

store/index.ts

```ts
import {configureStore} from "@reduxjs/toolkit";
import catReducer from "./catSlice.ts";

const store = configureStore({
    reducer: {
        cat: catReducer
    }
})

export type StoreType = ReturnType<typeof store.getState>

export default store;
```

main.ts

```tsx
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import {Provider} from "react-redux";
import store from "./store";

ReactDOM.createRoot(document.getElementById('root')!).render(
    <Provider store={store}>
        <App/>
    </Provider>
)
```

HelloCat.tsx

```tsx
import {useDispatch, useSelector} from "react-redux";
import {StoreType} from "../../store";
import {grow} from "../../store/catSlice.ts";

export default function HelloCat() {
    const cat = useSelector((store: StoreType) => store.cat)
    const dispatch = useDispatch()

    return <div>
        {cat.age}
        <button onClick={() => {
            dispatch(grow())
        }}>猫咪长大
        </button>
    </div>
}
```

## react-router使用方法

1. 从react-router-dom中引入createBrowserRouter方法和RouterProver组件
2. 使用createBrowserRouter创造router对象
3. 使用RouterProver组件代替根组件，router同名参数传入

```tsx
import * as React from "react";
import * as ReactDOM from "react-dom/client";
import {
    createBrowserRouter,
    RouterProvider,
} from "react-router-dom";
import "./index.css";

const router = createBrowserRouter([
    {
        path: "/",
        element: <div>Hello world!</div>,
    },
]);

ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
        <RouterProvider router={router}/>
    </React.StrictMode>
);
```

## React重复渲染解决方案

- 导致重复渲染的原因
    1. 在 React 中当组件的状态发生变化，就会重复渲染，这是 React 中组件更新的的内部机制，也是引起组件重复渲染的根本原因
    2. 当父组件重复渲染时，它的子组件都会跟着重新渲染。
    3. 当在使用 Context 时，如果 Context Provider 提供的 value 发生变化时，在所有使用 Context 数据的组件就会导致重复渲染，即使组件中只使用了Context 中的部分数据也会导致重复渲染
    4. 在组件中使用 hook 时，当 hook 中状态发生变化，会导致组件的重复渲染
- 避免重复渲染的方法
    1. 通过组合防止不必要的重复渲染
        - **不要在渲染函数中创建组件**
        - **move state down** 把经常由于状态变化而经常重复渲染的部分单独封装成子组件，避免影响其他组件
        - **components as props**把经常变的或者不变的部分提出来作为props传入
    2. 通过React.memo创造高阶组件避免不必要的重复渲染
       如果props不变化（可以用useMemo），就不会渲染组件
    3. 使用useMemo缓存context的数据
    4. 细分context的数据，变和不变的分别成两个context

## 什么是错误边界

应用中错误处理的一种方式, 是一个用于捕获后代组件错误，并渲染备用组件的 React
getDerivedStateFromError, componentDidCatch

```tsx
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = {hasError: false};
    }

    static getDerivedStateFromError(error) {
        // 更新 state，下一次渲染将展示备选 UI。
        return {hasError: true};
    }

    componentDidCatch(error, errorInfo) {
        console.log(error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            // 可以渲染任意自定义的备选 UI
            return <h1>出错啦！</h1>;
        }

        return this.props.children;
    }
}
```

对于函数组件，则可以用try catch进行错误捕获
对于服务端渲染，Suspense也可以作为错误边界

## 高阶组件（HOC）

HOC 是一个函数,它接受一个组件并返回一个新的组件,用于复用组件逻辑

## Portals

允许你将 JSX 作为 children 渲染至 DOM 的不同部分

```tsx
import {createPortal} from 'react-dom';

// ...

<div>
    <p>这个子节点被放置在父节点 div 中。</p>
    {createPortal(
        <p>这个子节点被放置在 document body 中。</p>,
        document.body
    )}
</div>
```

## Stack架构和Fiber架构

首先fiber它就具有多重含义:

- 一个框架
- 一种数据结构
- 动态的工作单元

react15 之前使用的是旧架构，也就是Stack架构，之后使用了新架构，也就是Fiber架构
Stack 架构存在的问题：

- CPU问题
  浏览器渲染画面时会被js执行堵塞，js执行时间过长会导致浏览器掉帧。
  stack架构模式虚拟dom是以树的结构组织的，当diff时需要递归对比，不支持打断，导致js执行时间过长。

- IO问题
  有些地方用户对相应速度要求非常高。比如输入文字，就要求不能卡顿。
  旧的架构模式在不支持优先级的调度

为解决旧架构的的问题，采用了新的fiber架构

1. 虚拟dom的实现方式由树改为了链表，child指向第一个子元素，sibling指向兄弟元素，return指向父元素
2. 协调过程从递归变成了可中断的循环，每次循环执行前可判断是否还有剩余时间
3. 新增调度器,根据任务进行优先级调度

新架构包括:

- Reconciler（协调器）：根据state变化计算UI变化
  在 React Fiber 架构中，每个 ReactElement 都有一个对应的 Fiber 节点，这个 Fiber 节点包含了更多的信息和更灵活的结构，以支持
  React 的高效更新和任务调度。

- Renderer（渲染器）：将UI渲染到宿主环境

- Scheduler（调度器）：调度任务的优先级，高优先级会先进入协调器

新架构渲染包括两个阶段：render 阶段和 commit 阶段

- render阶段：
  使用Reconciler（协调器）和Scheduler（调度器）
  生成一个workInProgress tree，部分节点标记了 side effects, side effects 描述了在下一个 commit 阶段需要插入、更新、删除、
  调用其生命周期方法等操作。
  这个阶段的特点是可以异步执行，中间的执行可以中断，可以根据可用时间来处理一个或多个 Fiber 节点，并且对用户来说是透明的

- commit:
  使用Renderer（渲染器）
  这个阶段会用到几个数据结构：
    - render 阶段生成 workInProgress tree
    - current tree 的 fiber 节点树，它直接用于更新UI
    - effects list，由 render 阶段生成的列表。

  这个阶段的任务是更新UI，并回调一些生命周期方法，包含以下一些操作：
    1. 在标记了 Snapshot effect 的节点上调用 getSnapshotBeforeUpdate 生命周期方法；
    2. 在标记了 Deletion effect 的节点上调用 componentWillUnmount 生命周期方法；
    3. 执行所有 DOM 插入，更新和删除；
    4. 将 workInProgress tree 树设置为 current 树；
    5. 在标记了 Placement effect 的节点上调用 componentDidMount 生命周期方法；
    6. 在标记了 Update effect 的节点上调用 componentDidUpdate 生命周期方法；

## 双缓冲

React 会存在两个 Fiber 树实例：current fiber tree和 workInProgress fiber tree 。

current fiber tree ：建立在第一个渲染器上，与 虚拟 DOM 具有一对一的关系，对应当前屏幕上显示内容，它的节点称为 current
fiber

workInProgress fiber tree ：正在内存中构建的 Fiber 树，即将用于渲染的树，它的节点称为 workInProgress fiber。

current fiber 与 workInProgress fiber 通过 alternate 属性连接

双缓存机制的好处:

- 可中断恢复
- 不会立即影响当前树，从而不会阻塞主线程的其他任务，比如用户的输入和交互

## 事件实现机制

react拥有自己的一套事件系统

- 合成事件对象   
  原生事件上的封装的对象，为了消除不同浏览器事件上的差异

- 事件传播机制    
  利用事件委托的原理，React基于FiberTree实现事件的捕获、目标和冒泡
    - 在 document 上绑定对应的事件监听
    - 触发事件时，event.target 找到触发的 element，并从上面拿到对应的 fiberNode
    - 从 fiber 向上收集所有绑定了该类型事件的 fiberNode 的事件回调
    - 反向遍历 list 并传递合成事件对象模拟 捕获
    - 正向遍历 list 并传递合成事件对象模拟 冒泡

好处：

- 跨平台兼容
- 更改事件名称为onXXX
- 不同事件有不同优先级
- 定制事件行为，onChange 改为 oninput
- 性能优化, 只在document上委托

## Hooks原理

- Hooks 函数在不同的情况下会执行不同的方法
  在render阶段渲染函数式组件时， 针对 Hook 会有三种策略
    - 首次渲染时，执行 HooksDispatcherOnMount 策略
    - 更新渲染时， 执行 HooksDispatcherOnUpdate 策略
    - 函数外渲染时，执行 ContextOnlyDispatcher 策略, 主要用于在不正确的执行上下文中进行报错提示，比如嵌套使用 hooks

- Hook 挂载
    - Hook 挂载策略执行会得到一个 Hook 对象
        - memoizedState： useState中 保存 state 信息 ｜ useEffect 中 保存着 effect 对象 ｜ useMemo 中 保存的是缓存的值和
          deps ｜ useRef 中保存的是 ref 对象。
        - baseQueue : usestate和useReducer中 保存最新的更新队列。
        - baseState ： usestate和useReducer中, 一次更新中 ，产生的最新state值。
        - queue ： 保存待更新队列 pendingQueue ，更新函数 dispatch 等信息。
        - next: 指向下一个 hooks对象。
    - 依次将 Hook 对象 连成一条链， WorkInProgress FiberNode 的 memoizedState 存储第一个 Hook 对象。

- Hook 更新
    - 从 Current FiberNode 中找到对应的 Hook 对象复用, 保证数据不会丢失。所以不能在循环中使用Hook，不然无法找到原先的 Hook
      对象
    - 


- useState
    - 挂载阶段
    - 更新阶段

## useState 和 useReducer

useState 就是 useReducer 的简化版本

- mount 阶段， useState 的 reducer 是 basicStateReducer，useReducer 是用户传入的
- update阶段，useState 直接调用的 updateReducer
