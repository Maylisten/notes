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
nestjs 使用文件系统路由![](https://raw.githubusercontent.com/Maylisten/image-hosting/main/202412042122159.png)![](https://raw.githubusercontent.com/Maylisten/image-hosting/main/202412042123347.png)
### Pages 和  Page
- `app` 根目录下的`pages.tsx`作为主页入口(`/`)，导出`<Page/>组件
- 每个文件夹下 `page.tsx` 导出`<Page/>`组件作为该路由的入口，也只有包含名为 `page.tsx`特殊文件的文件夹会被视为路由

### RootLayou 和 Layout
- `app`根目录下有一个`layout.tsx`文件用于导出 `<RootLayout />` 组件，其接收一个children属性，即 `pages.tsx`导出的`<Page/>`组件
- 每个目录下可以有一个`layout.tsx`文件用于导出 `<Layout />` 组件，其接收一个children属性，可以是同路径或下一路径的`<Page/>`、下一级`<Layout/>`。
- 在路由跳转时，只有页面部分会 re-render，layout 部分不会![image.png](https://raw.githubusercontent.com/Maylisten/image-hosting/main/202412042143178.png)
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
### 在服务端组件获取数据的优势
- 支持Promise，可以使用`async/await`语法，无需使用`useEffect`、`useState`或 data fetching 库
- 请求数据和逻辑运行在服务器上，只将结果发送到客户端，提升性能
- 直接查询数据库，无需额外的API层

### Fetch

#### 请求工具的选择
- 对于服务端组件请求，因为可以直接使用异步语法，所以推荐使用 nextjs 扩展的`fetch`函数
- 对于客户端组件请求，需要避免大量使用`useEffect`，所以推荐使用 SWR 或 React Query
#### unstable_cache 和 cache
- 使用 `unstable_cache` 缓存（Data Cache）响应，可以让页面在下次 build 时预加载（静态渲染策略）
	```tsx
	import { unstable_cache } from 'next/cache'
	import { db, posts } from '@/lib/db'
	 
	const getPosts = unstable_cache(
	  async () => {
	    return await db.select().from(posts)
	  },
	  ['posts'],
	  { revalidate: 3600, tags: ['posts'] }
	)
	 
	export default async function Page() {
	  const allPosts = await getPosts()
	 
	  return (
	    <ul>
	      {allPosts.map((post) => (
	        <li key={post.id}>{post.title}</li>
	      ))}
	    </ul>
	  )
	}
	```
- 使用`cache`缓存响应，可以避免单次渲染的重复请求（Request Memoization）
	```tsx
	import { cache } from 'react'
import { db, posts, eq } from '@/lib/db' // Example with Drizzle ORM
import { notFound } from 'next/navigation'
 
export const getPost = cache(async (id) => {
  const post = await db.query.posts.findFirst({
    where: eq(posts.id, parseInt(id)),
  })
 
  if (!post) notFound()
  return post
})
```
#### Preloading Data
利用缓存，在渲染组件前提前获取数据起到缓存效果
```tsx
// components/Item.tsx
import { getItem } from '@/utils/get-item'
 
export const preload = (id: string) => {
  // void evaluates the given expression and returns undefined
  // https://developer.mozilla.org/docs/Web/JavaScript/Reference/Operators/void
  void getItem(id)
}
export default async function Item({ id }: { id: string }) {
  const result = await getItem(id)
  // ...
}

// app/item/[id]/page.tsx
import Item, { preload, checkIsAvailable } from '@/components/Item'
 
export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  // starting loading item data
  preload(id)
  // perform another asynchronous task
  const isAvailable = await checkIsAvailable()
 
  return isAvailable ? <Item id={id} /> : null
}
```

#### server-only
通过引入 server-only 库可以保证代码运行在服务端（或客户端），否则抛出异常
```tsx
import { cache } from 'react'
import 'server-only'
 
export const preload = (id: string) => {
  void getItem(id)
}
 
export const getItem = cache(async (id: string) => {
  // ...
})
```

### Server Action
#### 基本概念
- Server Action 是运行在服务端的异步函数
- 如果在服务端组件使用，即使浏览器不支持 JavaScript 也可以起作用
- 可以用在服务端或客户端组件中
- 没使用限制
- 底层创建了一个公共的 HTTP POST端点，并创建加密的、非确定性（build 的时候会重新计算）的 ID，以便客户端可以引用和调用服务器操作
- 会使用 nextjs 的缓存机制
- 参数和返回值必须可以被 React 序列化
- 继承被使用时Page 或 Layout 的 runtime 和 Route Segment Config
#### 'use server'
'use server' 可以使用在文件顶部或者在函数顶部
```tsx
'use server'

export default function Page() {
  // Server Action
  async function create() {
    'use server'
    // Mutate data
  }
 
  return '...'
}
```

#### 使用限制
Server Action 可以在 Client Component 中处理表单提交、事件处理、useEffect或者被当做参数传递，没有任何限制

#### 基本使用
```tsx
// app/ui/signup.tsx
'use client'
 
import { useActionState } from 'react'
import { createUser } from '@/app/actions'
 
const initialState = {
  message: '',
}
 
export function Signup() {
  const [state, formAction, pending] = useActionState(createUser, initialState)
 
  return (
    <form action={formAction}>
      <label htmlFor="email">Email</label>
      <input type="text" id="email" name="email" required />
      {/* ... */}
      <p aria-live="polite">{state?.message}</p>
      <button disabled={pending}>Sign up</button>
    </form>
  )
}

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
```

#### 闭包
使用 Server Action 时可能会产生闭包
```tsx
export default async function Page() {
  const publishVersion = await getLatestVersion();
 
  async function publish() {
    "use server";
    if (publishVersion !== await getLatestVersion()) {
      throw new Error('The version has changed since pressing publish');
    }
    ...
  }
 
  return (
    <form>
      <button formAction={publish}>Publish</button>
    </form>
  );
}
```
这会导致数据从客户端流回服务端，为了保护数据安全，nextjs 会在构建时产生额外的密钥，推荐使用 React taint APIs 保护数据隐私

### ISR（Incremental Static Regeneration）
#### 定义
在构建时生成初始的静态页面，同时允许在运行时按需重新生成页面内容（缓存动态渲染结果，变为静态渲染），特征如下：
- 更新静态内容而不需要重建整个网站（因为是增量更新，请求后的页面会被缓存下来）
- 通过为大多数请求提供预渲染的静态页面来减少服务器负载（因为更新后变成了静态渲染）
- 确保自动添加适当的 `cache-control` 头部到页面
- 处理大量内容页面而无需长时间 `next build` 时间（因为增量更新）

#### 使用方法
```tsx
interface Post {
  id: string
  title: string
  content: string
}
 
// 重新校验数据的间隔时间
// 0： dynamic render
// 30: 30s 校验一次
// false（默认）: 无期限缓存，不校验
export const revalidate = 60

// 是否有动态的 Param
// false: 访问没缓存的页面会 404
// true： 访问没缓存的页面会动态渲染
export const dynamicParams = true 
 
export async function generateStaticParams() {
  const posts: Post[] = await fetch('https://api.vercel.app/blog').then((res) =>
    res.json()
  )
  return posts.map((post) => ({
    id: String(post.id),
  }))
}
 
export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const id = (await params).id
  const post: Post = await fetch(`https://api.vercel.app/blog/${id}`).then(
    (res) => res.json()
  )
  return (
    <main>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
    </main>
  )
}
```

#### Revalidate
- 基于时间
```tsx
'use server'
import { revalidatePath } from 'next/cache'
export async function createPost() {
// Invalidate the /posts route in the cache
revalidatePath('/posts')
}
```
	
 - revalidatePath                                                                 
```tsx
'use server'
 
import { revalidatePath } from 'next/cache'
 
export async function createPost() {
  // Invalidate the /posts route in the cache
  revalidatePath('/posts')
}
```

 - revalidateTag                                
```tsx
/// app/blog/page.tsx
import { unstable_cache } from 'next/cache'
import { db, posts } from '@/lib/db'
 
const getCachedPosts = unstable_cache(
  async () => {
	return await db.select().from(posts)
  },
  ['posts'],
  { revalidate: 3600, tags: ['posts'] }
)
 
export default async function Page() {
  const posts = getCachedPosts()
  // ...
}

/// app/actions.ts
'use server'

import { revalidateTag } from 'next/cache'

export async function createPost() {
// Invalidate all data tagged with 'posts' in the cache
revalidateTag('posts')
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

| Dynamic APIs | Data       | Route                |
| ------------ | ---------- | -------------------- |
| No           | Cached     | Statically Rendered  |
| Yes          | Cached     | Dynamically Rendered |
| No           | Not Cached | Dynamically Rendered |
| Yes          | Not Cached | Dynamically Rendered |

- 静态策略（默认）
	- 在 build 的时候 render **页面**，或者当 data revalidation 的时候在后台 render
	- 可以跨用户、跨请求共享render 结果，构建结果可以放置到边缘服务器中
	- 要求使用的数据在构建时已知，并且需要在 Data Cache 中
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
	-  在层级尽可能深的地方使用自定义的上下文 Provider
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
### 缓存类型

| Mechanism           | What                       | Where  | Purpose                                         | Duration                        |
| ------------------- | -------------------------- | ------ | ----------------------------------------------- | ------------------------------- |
| Request Memoization | Return values of functions | Server | Re-use data in a React Component tree           | Per-request lifecycle           |
| Data Cache          | Data                       | Server | Store data across user requests and deployments | Persistent (can be revalidated) |
| Full Route Cache    | HTML and RSC payload       | Server | Reduce rendering cost and improve performance   | Persistent (can be revalidated) |
| Router Cache        | RSC Payload                | Client | Reduce server requests on navigation            | User session or time-based      |


### 缓存影响因素
![image.png](http://43.142.166.50:9001/image-hosting/20241212093025432.png)
- 静态渲染/动态渲染
- 已有缓存/无缓存
- 初次访问/后续导航
### Request Memoization
#### 基本定义
 React 扩展了 fetch API，自动缓存具有相同 URL 和选项的请求。![image.png](http://43.142.166.50:9001/image-hosting/20241210214602477.png)
#### 有效时期
缓存从服务请求直到 React 组件树完成渲染
#### 刷新方法(revalidate)
无需刷新
#### 禁用方法
不使用 GET 方法
#### 重点总结
- 在一次组件树的 render 过程中只会执行一次， 利用这一点可以方便进行不同页面的共享
- 一旦路由渲染完成且渲染过程结束，Request Memoization 会被重置清除
- Request Memoization 优先级高于 Data Cache
- Request Memoization **只作用于 Server Component**，Route Handler 和 Client Component 中没有
-  Request Memoization **只对 GET 有效**，POST、DELTE 等请求方法无效

### Data Cache
#### 基本定义
Next.js 扩展了原生的 fetch API，内置了一个可以跨请求和部署持久化的 Data Cache，并允许服务器上的每个请求设置自己的持久化缓存语义。
nextjs 中的 `cache`和 http 中的`cache`含义不同。在浏览器中，`fetch` 的 `cache` 选项表示请求如何与浏览器的 HTTP 缓存交互。但是在 Next.js 中，`cache` 选项表示服务器端请求如何与服务器端的数据缓存交互。
![image.png](http://43.142.166.50:9001/image-hosting/20241210222212692.png)
#### 有效时期
除非 revalidate 或者禁止使用 ，Data Cache 一直存在，并且跨入站请求和部署共享。
#### 刷新方法(revalidate)
- 基于时间
	开发者可以设置 Data Cache 的自动刷新的时间，*在数据刷新过程中或者刷新数据失败时，Data Cache 中会保留旧的数据*
	```tsx
	// Revalidate at most every hour
	fetch('https://...', { next: { revalidate: 3600 } })
	```
	![image.png](http://43.142.166.50:9001/image-hosting/20241210222729445.png)
- 基于需求
	按照实际需求，开发者可以主动去更新Data Cache，实现方式是*清除 Data Cache 中的数据*，在下次请求时自然会重新请求。常用在 `Route Handler` 和 `Server Action`中![image.png](http://43.142.166.50:9001/image-hosting/20241211145149502.png)
	- revalidatePath
		手动重新验证数据，并 re-render 指定的路由
		```tsx
		revalidatePath('/')		
		```
	- revalidateTag
		创建 Data Cache 时可以为条目添加一个或多个标签，之后可以调用`revalidateTag`来清除与该标签关联的 Data Cache
		```tsx
		/// Cache data with a tag
		fefetch(`https://...`, { next: { tags: ['a', 'b', 'c'] } })
		
		//// Revalidate entries with a specific 
		tagrevalidateTag('a')
		`````
#### 禁用方法
```tsx
let data = await fetch('https://api.vercel.app/blog', { cache: 'no-store' })
```

#### 重点总结
- 基于时间的刷新如果失败不会清空旧的数据,而是在下次请求的时候重新尝试获取新数据；基于需求的主动刷新通过清空数据来实现数据的刷新
- `revalidatePath`先清空Data Cache，后 re-render
- 就算重新部署，Data Cache 也不会清除，是跨部署共享的

### Full Route Cache
#### 基本定义
在 Static Render 或 revalidation 时，Next.js 会在服务端缓存通过 Render 得到的 RSC Payload 和 HTML
![image.png](http://43.142.166.50:9001/image-hosting/20241211162137719.png)

#### 有效时间
永久有效
#### 刷新方法(revalidate)
- revalidation 或禁用 Data Cache 会使 Full Route Cache 刷新（因为 render 依赖数据，所以 Nextjs 会在 Data Cache 更新时同步更新 Full Route Cache）
- 重新部署（不同于 Data Cache)
#### 禁用方法
禁用 Full Router Cache 的方法本质就是将静态渲染改为动态渲染的方法
- 使用 Dynamic API 
- 禁用 Data Cache
- 使用`dynamic = 'force-dynamic'` 或 `revalidate = 0` route 配置选项
#### 重点总结
- 只有静态渲染会使用 Full Route Cache，动态渲染会直接略过
- Full Route Cache 是依赖 Data Cache 的，如果在 Render 时取得的数据没有被缓存，则 Full Route Cache 也会失效

### Client Route Cache
#### 基本定义
在客户端内存中，通过存储 RSC payload ，缓存访问过的路由和预加载的路由
- Layout 缓存，用于部分渲染策略
- Loading 预加载，用于立即渲染
- Page 默认不预加载, 但实际上在前进后退跳转时会使用，未来会有相关功能（ `staleTimes` 实验特性）
![image.png](http://43.142.166.50:9001/image-hosting/20241211193814495.png)
#### 有效时间
缓存存储在浏览器的临时内存中
- 会话
	页面刷新时丢失全部的 Client Route Cache
- Automatic Invalidation Period
	- 默认预加载 (`prefetch={null}` 或未指定)：动态页面不缓存，静态页面缓存5分钟
	- 完全预取预加载(`prefetch={true}` 或 `router.prefetch`)：静态和动态页面均缓存为5分钟
#### 刷新方法(revalidate)
 - 在 Server Action 中使用 `revalidatePath`和 `revalidateTag`
 - 使用 `router.refresh`
#### 禁用方法
- Page 默认就是不使用 Client Route Cache
- 通过配置`prefetch=false`阻止`<Link/>`预加载

#### 重点总结
- 无论是 Full Route Cache 还是 Client Route  Cache 存储的都是服务端渲染的结果
- 动态渲染和静态渲染都会使用 Client Route Cache，并且比 Full Route Cache 优先级更高
- 页面刷新会一次性清空所有的 Client Route Cache，而自动失效时间计算是各个路由段相互独立的
- Prefetch 时默认缓存的是 Loading 而不是 Page

### Data Cache 和 Route Cache
#### Data Cache and Full Route Cache
- revalidation 或 禁用 Data Cache 的行为会使得 Full Route Cache 失效，因为 Render 依赖于数据
- 取消 Full Route Cache 不会影响 Data Cache。动态渲染可以同时使用缓存和没缓存的数据，重新渲染也不会导致缓存的数据重新获取
#### Data Cache and Client-side Router cache
- 在 Server Action 中，可以使用`revalidatePath`或`revalidateTag`立即清除Data Cache 和 Client Router Cache
- 在 Route Handler 中，因为 Route Handler 与路由无关，更新或取消 Data Cache 不会立刻是清除 Router Cache，除非刷新网页或者 Client Router Cache 过期