# Nextjs 学习笔记

## Style 工具库

| tailwindcss<br> | 原子类型                      |
| --------------- | ------------------------- |
| cssModule<br>   | 引入 css                    |
| CSS-in-JS<br>   | 直接在 jsx 中使用style 标签       |
| clsx            | 连接多个 class 到一个 class 字符串中 |


## 字体和图片
### 自定义字体优化
nextjs 会在构建时同时打包字体，避免客户端单独请求字体文件
使用方法：
- 在`app/ui`下创建`fonts.ts`文件用于管理自己定义字体
```ts
import {Inter, Lusitana} from 'next/font/google';  
export const inter = Inter({subsets: ['latin']});  
export const lusitana = Lusitana({subsets: ['latin'], weight: "400"});
```
- 在`tsx` 文件中引入使用
```tsx
import '@/app/ui/global.css';
import { inter } from '@/app/ui/fonts';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} antialiased`}>{children}</body>
    </html>
  );
}
```

### 图片优化
使用 `<Image/>` 组件
- 当图片加载时自动防止布局偏移。
- 调整图片大小以避免向视口较小的设备发送大图片
- 默认启用懒加载图片（图片在进入视口时加载）
- 自动支持 [WebP](https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Image_types#webp) 、 [AVIF](https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Image_types#avif_image) 等现代格式图片的下载（浏览器支持的条件下）

## 项目结构
### 文件系统路由
nestjs 使用文件系统路由
![](https://raw.githubusercontent.com/Maylisten/image-hosting/main/202412042122159.png)
![](https://raw.githubusercontent.com/Maylisten/image-hosting/main/202412042123347.png)
### Pages 和  Page
- `app` 根目录下的`pages.tsx`作为主页入口(`/`)，导出`<Page/>组件
- 每个文件夹下 `page.tsx` 导出`<Page/>`组件作为该路由的入口，也只有包含名为 `page.tsx`特殊文件的文件夹会被视为路由

### RootLayou 和 Layout
- `app`根目录下有一个`layout.tsx`文件用于导出 `<RootLayout />` 组件，其接收一个children属性，即 `pages.tsx`导出的`<Page/>`组件
- 每个目录下可以有一个`layout.tsx`文件用于导出 `<Layout />` 组件，其接收一个children属性，可以是同路径或下一路径的`<Page/>`、下一级`<Layout/>`。
- 在路由跳转时，只有页面部分会 re-render，layout 部分不会

![image.png](https://raw.githubusercontent.com/Maylisten/image-hosting/main/202412042143178.png)
### loading
`loading.tsx`是一个基于 `<Suspense/>` 构建的 Next.js 特殊文件，允许在页面内容加载时创建一个备选 UI 来显示

### Route Group 和 Private
- **Route Group**:  名称用`()`包裹的**文件夹自身**不会出现在路由中
- **Private**: 名称加上`_`前缀的**文件夹以及子文件夹**不会出现在路由中

### Template
- 会夹在 layout 和 page 之间
- 和 layout 类似，但是每次切换路由时，会重新挂在 DOM，state 不会保留

### 共享
共享文件夹没有明确命名规范，可以创建`src/components`放置共享组件、创建`src/lib`放置共享模块（方法）

## 路由
### 跳转
- `<Link/>` 『client』有默认的预加载，推荐使用
```jsx
import Link from 'next/link'
 
export default function Page() {
  return <Link href="/dashboard">Dashboard</Link>
}	
```
- useRouter()『client』
```jsx
'use client'
 
import { useRouter } from 'next/navigation'
 
export default function Page() {
  const router = useRouter()
 
  return (
    <button type="button" onClick={() => router.push('/dashboard')}>
      Dashboard
    </button>
  )
}
```
- redirect 『server』
```jsx
import { redirect } from 'next/navigation'
 
async function fetchTeam(id: string) {
  const res = await fetch('https://...')
  if (!res.ok) return undefined
  return res.json()
}
 
export default async function Profile({ params }: { params: { id: string } }) {
  const team = await fetchTeam(params.id)
  if (!team) {
    redirect('/login')
  }
 
  // ...
}
```

### Code Split
定义：将程序代码拆分成更小bundle，只传输需要的部分，从而减少请求时间。
Nextjs 中 Server Component 自动根据路由端进行 Code Split

### Prefetching
定义：用户访问前，在后台预加载路由页面，在 nextjs 中只有生产环境会有预加载
- `<Link/>`出现在视口中时，默认自动一路向下预加载到子树下的第一个 loading 为止，并缓存30s
- `router.prefetch()` 可以手动预加载

### Partial Rending
定义：导航中，只有路由发生变化的部分会re-render，其他部分状态不变



## Error
#### Server Actions
使用 useFormState 管理 ServerAction 的状态，包括错误
```jsx
// app/actions.ts
'use server'
 
import { redirect } from 'next/navigation'
 
export async function createUser(prevState: any, formData: FormData) {
  const res = await fetch('https://...')
  const json = await res.json()
 
  if (!res.ok) {
    return { message: 'Please enter a valid email' }
  }
 
  redirect('/dashboard')
}

// app/ui/signup.tsx
'use client'
 
import { useFormState } from 'react-dom'
import { createUser } from '@/app/actions'
 
const initialState = {
  message: '',
}
 
export function Signup() {
  const [state, formAction] = useFormState(createUser, initialState)
 
  return (
    <form action={formAction}>
      <label htmlFor="email">Email</label>
      <input type="text" id="email" name="email" required />
      {/* ... */}
      <p aria-live="polite">{state?.message}</p>
      <button>Sign up</button>
    </form>
  )
}
```

### Server Component Error
- 返回错误对应的 UI
- 使用 redirect 函数重定向

### Error Bounding
- 使用 error.tsx
- 使用`<ErrorBounding/>`组件

## Data Query
### 服务端组件
#### 服务端组件获取数据的优势
- 支持Promise，可以使用`async/await`语法，无需使用`useEffect`、`useState`或 data fetching 库
- 请求数据和逻辑运行在服务器上，只将结果发送到客户端，提升性能
- 直接查询数据库，无需额外的API层

#### 使用 nestjs sdk 和 sql 进行查询
- 服务端组件使用异步组件，并在 render 函数中直接查询
```tsx
import { Card } from '@/app/ui/dashboard/cards';
import RevenueChart from '@/app/ui/dashboard/revenue-chart';
import LatestInvoices from '@/app/ui/dashboard/latest-invoices';
import { lusitana } from '@/app/ui/fonts';
import { fetchRevenue } from '@/app/lib/data';
 
export default async function Page() {
	const revenue = await fetchRevenue();
	return (
	<main>
	  <h1 className={`${lusitana.className} mb-4 text-xl md:text-2xl`}>
		Dashboard
	  </h1>
	  <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
	  </div>
	  <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-4 lg:grid-cols-8">
		<RevenueChart revenue={revenue}  />
	  </div>
	</main>
	);
}
```
- 使用sql 语句进行异步数据库查询
```ts
export async function fetchRevenue() {  
  try {  
    const data = await sql<Revenue>`SELECT * FROM revenue`;  
    return data.rows;  
  } catch (error) {  
    console.error('Database Error:', error);  
    throw new Error('Failed to fetch revenue data.');  
  }  
}
```

## 渲染

### 什么是服务端渲染
1. 在服务端，渲染任务会根据路由片段和`Suspense Boundaries`分成不同的 chunk，对每一个 chunk：
	1. React渲染 Server Component 得到**React Server Component Payload (RSC Payload)**
	2. Next.js 使用 RSC Payload 和客户端组件 JavaScript 指令在服务器上渲染 HTML

2. 在客户端：
	1. 将服务端渲染的 HTML 用于初始页面加载，提供快速的无交互的预览
	2. 使用 RSC Payload 协调 Server 和 Client 组件树，并更新 DOM 树
	3. 使用 JavaScript 指令`hydrate`DOM 树和客户端组件，使应用程序具有交互性

### RSC Payload
RSC Payload 是 React 服务器组件树的紧凑二进制数据结构， React在客户端用它来更新真实 DOM。
- 服务器组件的渲染结果
- **客户端组件应渲染的位置占位符**以及对**其 JavaScript 文件的引用**
- 从服务器组件传递给客户端组件的所有 props

### Hydration
`Hydration`是将事件监听器附加到DOM上，使静态HTML变的可交互的过程。在后台，`Hydration`是通过 hydrateRoot React API 完成的。
### 服务端组件优点
- 数据获取更加快速和安全
- 跨用户缓存内容
- 加快首屏加载
- 搜索引擎优化
- 将渲染工作分成块，准备好后将其流式传输到客户端。使用户可以提前看到页面的一部分，而无需等待整个页面在服务器上渲染完成

### 服务端渲染策略
关于服务端渲染，开发者往往不需要考虑具体选择哪个方案，这些 nextjs 框架会根据具体情况自动完成，而是应该考虑将哪些部分需要用`<Suspense/>`包裹，什么时候使用 `cache` 或者 `revalidate specific data`
- 静态策略（默认）
	- 在 build 的时候 render **页面**，或者当 data revalidation 的时候在后台 render
	- 可以跨用户、跨请求共享render 结果，构建结果可以放置到边缘服务器中
	- 要求使用的数据在构建时已知
- 动态策略
	- 每次响应请求的时候 render **页面**
	- 适用于使用的数据和请求和具体场景有关（构建时未知）
	- 动态策略也可以有缓存，因为  RSC Payload 和 fetch 请求的 data 是分开存储的
- 流
	- 让用户可以将 render 工作拆分成不同的 chunk，服务端逐步的将 render 好的结果传递到客户端
	- nextjs 默认在 App Router 中自动开启流策略
	- 提升用户体验和首屏相应速度
	- 可以通过 loading.tsx 触发路由级别的流策略，使用 React Suspense 触发组件级别的流策略
- 部分渲染（*实验功能*）
	- 使用 Suspense boundary 可以隔离需要动态渲染的组件，使得页面其他部分可以使用效率更高的静态渲染策略
### 服务端组件使用技巧
- 共享数据
	通过`fetch`(nextjs 提供的)和 cache 共享获取的数据
- 强制规定渲染环境
	使用`server-only`库强制模块只能运行在特定环境，否则打包时报错
- 使用第三方模块
	一些第三方库和组件使用了钩子和 windows API，需要显式声明需要在客户端环境下渲染
```tsx
'use client'
 
import { Carousel } from 'acme-carousel'
 
export default Carousel
```
- 使用 context
	- 在单独的文件中创建自定义上下文 Provider
	-  在层级尽可能深的地方使用自定义的上下文 Provider
```tsx
'use client'
 
import { createContext } from 'react'
 
export const ThemeContext = createContext({})
 
export default function ThemeProvider({
  children,
}: {
  children: React.ReactNode
}) {
  return <ThemeContext.Provider value="dark">{children}</ThemeContext.Provider>
}
```

```tsx
import ThemeProvider from './theme-provider'
 
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html>
      <body>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  )
}
```

### 客户端组件
交互式UI，在服务器上预渲染，使用客户端 JavaScript 在浏览器中运行
客户端组件引入的模块和子组件都运行在浏览器端

### 客户端组件优点
- 交互性: 客户端组件可以使用状态、效果和事件监听器，这意味着它们可以向用户提供即时反馈并更新用户界面
- 浏览器 API: 客户端组件可以访问浏览器 API

### 客户端边界
- 数据只能**从服务端流向客户端**，即：客户端组件可以是服务端组件的子组件，而服务端组件不能是客户端组件的子组件
- APP Router 默认组件为服务端组件，可以使用"use client" 标志渲染边界，渲染边界以下的组件和模块都运行在客户端

### 客户端渲染策略
- 整页加载时
	整页加载和服务端渲染过程基本一致：
	1. 服务端先通过 RSC Payload （内包含客户端组件占位）和 客户端组件的 js 指令得到静态 html
	2. 客户端先直接展示 html
	3. 客户端使用 RSC Payload `reconcile` 客户端组件树和服务端组件树，然后更新 DOM
	4. 客户端水合，使 UI 变成响应式的
- 部分加载时，完全在客户端渲染，没有服务器预渲染的 HTML
	1. 客户端下载 JavaScript bundle
	2. 客户端使用 RSC Payload `reconcile` 客户端组件树和服务端组件树，然后更新 DOM

### 客户端组件使用技巧
- 将客户端组件下放
	客户端组件在组件树越底层越好，可以减小客户端组件的 bundle size
- 将数据从服务端组件传给客户端组件 

## Cache
