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
使用 error.tsx

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

### 服务端组件
编写可在服务器上渲染并可选择性缓存的UI，在组件文件开头加上`use server`，组件默认为服务端渲染
- 数据获取更加快速和安全
- 跨用户缓存内容
- 加快首屏加载
- 搜索引擎优化
- 将渲染工作分成块，准备好后将其流式传输到客户端。使用户可以提前看到页面的一部分，而无需等待整个页面在服务器上渲染完成

### SSR （服务端渲染）过程
1. 在服务端，渲染任务会根据路由和『Suspense 边界』分成不同的 chunk，对每一个 chunk：
	1. React将服务器组件渲染成一种特殊的数据格式，称为**React Server Component Payload (RSC Payload)**
	2. Next.js 使用 RSC Payload 和客户端组件 JavaScript 指令在服务器上渲染 HTML

2. 在客户端：
	1. 将服务端渲染的 HTML 用于初始页面加载，提供快速的无交互的预览
	2. 使用 RSC Payload 协调 Server 和 Client 组件树，同步更新 DOM 树
	3. 使用 JavaScript 指令`hydrate`DOM 树和客户端组件，使应用程序具有交互性

### React Server Component Payload (RSC Payload)
是已经渲染好的 React Server Components 树的二进制表现形式，用于React 在客户端更新 DOM，其具体内容包括：
- 所有 Server Component 的渲染结果
- 用于指定客户端组件应渲染的位置的占位符，以及其对应的 JavaScript 文件引用
- 所有从 Server Component 传给 Client Component 的 props

### 服务端渲染策略


### 客户端组件


## Cache