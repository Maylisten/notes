//
// export default [
//     {
//         text: "面试",
//         items: [
//             {text: "精选", link: "/interview/精选/handpick"},
//             {text: "HTML", link: "/interview/html/HTML 面试题汇总"},
//             {text: "CSS", link: "/interview/css/CSS 面试题汇总.md"},
//             {text: "JS", link: "/interview/javascript/javascript 面试题汇总 "},
//             {text: "promise", link: "/interview/promise/Promise面试题归总"},
//             {text: "工程化", link: "/interview/工程化/工程化面试题汇总"},
//             {text: "网络", link: "/interview/网络/网络面试题汇总"},
//             {text: "浏览器", link: "/interview/浏览器/浏览器面试题汇总"},
//             {text: "框架", link: "/interview/框架/框架面试题汇总"},
//             { text: "Vue", link: "/interview/vue/Vue笔面试题汇总" },
//             { text: "React", link: "/interview/react/React面试题汇总" },
//         ],
//     },
//     {
//         text: "threejs",
//         items: [
//             {text: "加载倾斜摄影", link: "/three/倾斜摄影/threejs加载倾斜摄影"},
//         ],
//     },
//     {
//         text: "docker",
//         items: [
//             {text: "docker安装", link: "/docker/安装/docker安装"},
//             {text: "常用命令", link: "/docker/命令/docker常用命令"},
//         ],
//     },
// ];

import AutoNavPlugin from "vitepress-auto-nav-sidebar";

export const {nav, sidebar} = AutoNavPlugin({
  entry: "./",
  ignoreFiles: ['index.md'],
  ignoreFolders: [".git", ".github", ".idea", ".obsidian", ".vitepress", "node_modules"],
  showNavIcon: false,
  showSideIcon: false,
})
