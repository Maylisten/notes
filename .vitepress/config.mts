import {defineConfig} from "vitepress";
import {nav, sidebar} from "./menu";

export default defineConfig({
  title: "徐瀚的前端学习笔记",
  description: "一个励志成为前端大佬的学习笔记",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav,
    sidebar,
    socialLinks: [
      {icon: "github", link: "https://github.com/vuejs/vitepress"},
    ],
    search: {
      provider: "local",
    },
  },
  base: "/blog/",
});
