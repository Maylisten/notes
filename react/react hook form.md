# React Hook Form

## 使用意义
- 管理表单数据（state）
- 表单校验
- 提交表单

## 安装
```shell
npm install react-hook-form
```

## 管理表单数据
1. 调用`useForm`钩子并从中解构得到`register`、`handleSubmit`
2. `register`方法调用会返回一系列属性（`value`、`onChange`、`onBlur`...），直接解构传入到 `input`组件中
3.  使用 `handleSubmit` 处理表单的提交逻辑（会阻止表单的默认提交行为）和错误

```tsx
"use client"  
import {useForm,FieldErrors} from "react-hook-form";  
import {DevTool} from "@hookform/devtools";  
  
type FormValues = {  
  name: string,  
  social: {  
    facebook: string,  
    twitter: string  
  }
}  
  
const defaultFormValues: FormValues = {  
  name: "11",  
  social: {  
    facebook: "11",  
    twitter: "11"  
  }
}  
  
function YoutubeForm() {  
  const form = useForm<FormValues>({  
    defaultValues: defaultFormValues  
  });  
  const {register, handleSubmit} = form;  
 
  const onSubmit = (data: FormValues) => {  
    console.log(data)  
  }  
  
  const onError = (error: FieldErrors<FormValues>) => {  
	console.log(error)  
  }
  
  return (  
    <div className="w-full h-full flex flex-col justify-center items-center gap-4">  
      <div className="text-xl font-bold"> YouTube 表单</div>  
      <form className=" w-[300px] h-fit flex flex-col gap-2" onSubmit={handleSubmit(onSubmit)} >  
        <div className="form-item">  
          <label htmlFor="name">Name</label>  
          <input type="text" id="name" className="border" {...register("name")}/>  
        </div>  

        <div className="form-item">  
          <label htmlFor="facebook">Facebook</label>  
          <input type="text" id="facebook" className="border"  {...register("social.facebook")}/>    
        </div>  
  
        <div className="form-item">  
          <label htmlFor="twitter">Twitter</label>  
          <input type="text" id="twitter" className="border"  {...register("social.twitter")}/>  
        </div>  
        
        <Button type="submit" className="mt-2">Submit</Button>  
      </form>  
    </div>  
  );  
}  
  
export default YoutubeForm;
```

## 表单校验
1. `form`表单添加`noValidate`属性组织默认校验行为
2. 解构出 `error`、`isDirty`、`isValid`对象
```tsx
const form = useForm<FormValues>({  
    defaultValues: defaultFormValues  
});
const {formState} = form;  

const {errors,isDirty,isValid} = formState;
```
3. `register`第二个参数添加`required`/`pattern`/`validate`参数，并可视化错误信息
```tsx
<input type="email" id="email" className="border"  {...register("email", {  
  required: "please enter a valid email address",  
  pattern: {  
    value: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,  
    message: "email is wrong"  
  },  
  validate: {  
    notAdmin: (fieldValue) => {  
      return fieldValue !== "xuhan57@qq.com" || "enter a different email address of admin"  
    }  
  }  
})}/>
<p className="text-red-400">{errors.email?.message}</p>
```
4. 限制 submit 按钮
```tsx

<Button type="submit" className="mt-2" disabled={!isDirty || !isValid}>Submit</Button>

```
## 主动获取表单数据
- 使用 watch
```tsx

const watchName = watch("name")

```
- 使用 getValue 获取数据
```tsx

const form = useForm<FormValues>({  
defaultValues: defaultFormValues  
});
const {getValues} = form;
getValue("name")

```

## 动态表单
- 初始化数组数据结构
```tsx
type FormValues = {  
  phoneNumbers: { number: string }[]
}  
  
const defaultFormValues: FormValues = {  
  phoneNumbers: [{number: "11"}]
}
```
- 使用`useFieldArray`钩子解构得到`fields`、`append`、`remove`
```tsx
const form = useForm<FormValues>({  
  defaultValues: defaultFormValues  
});  
const { control } = form;  
  
const {fields, append, remove} = useFieldArray({  
  name: "phoneNumbers",  
  control  
})
```
- 可视化表单
```tsx
 <div className="w-full flex flex-col gap-2">  
    {fields.map((field, index) => {  
      return (  
        <div className="w-full flex flex-col gap-1" key={field.id}>  
          <div className="w-full flex flex-row justify-between gap-2">  
            <input type="text" className="border w-full"  
                   {...register(`phoneNumbers.${index}.number`, {  
                     required: "phone number is required"  
                   })}/>  
            {
            index > 0 && 
	            (<Button onClick={() => remove(index)}> 删除 </Button>)  
            }  
          </div>  
          <p className="text-red-400">{errors.phoneNumbers?.[index]?.number?.message}</p>  
        </div>  
      )  
    })}  
  </div>  
</div>
  
<Button type="button" className="mt-2" onClick={() => append({number: ""})}>ADD PHONE NUMBER</Button>
```

## disabled
disabled 会同时禁用输入和校验
```tsx
<div className="form-item">  
  <label htmlFor="password">Password</label>  
  <input type="password" id="password" className="border"  {...register("password", {  
    required: "Please enter a password"  
  })}/>  
  <p className="text-red-400">{errors.password?.message}</p>  
</div>  
  
<div className="form-item">  
  <label htmlFor="confirmPassword">Password</label>  
  <input type="password" id="confirmPassword" className="border"  {...register("confirmPassword", {  
    required: "Please enter a confirmPassword",  
    disabled: watch("password") === "",  
    validate: {  
      onCheck: (_, formValues) => {  
        return formValues.password === formValues.confirmPassword || "Two password entries are inconsistent."  
      }  
    }  
  })}/>  
  <p className="text-red-400">{errors.confirmPassword?.message}</p>  
</div>
```

## 获取 Dirty 和  Touched 信息
Dirty 指的是表单数据是否被用户修改过
Touched 指的是用户是否触发过表单的交互行为
Dirty 和 Toched 是每个表单项都拥有且相互独立的属性
```tsx

const form = useForm<FormValues>({  
    defaultValues: defaultFormValues  
});
const {formState} = form;  
// 此处的 isDirty 是对整个表单而言的
const {errors, touchFields,dirtyFields,isDirty} = formState;

```

## 表单状态
- `isValid`：是否全部校验通过
- `isDirty`：是否有数据发生改变
- `isSubmitting`：是否处于提交过程中
- `isSubmitted`：是否已经提交完成
- `isSubmitSuccessful`：是否提交成功
- `submitCount`：提交次数
```tsx

const {isValid, isDirty, isSubmitting, isSubmitted, isSubmitSuccessful,submitCount} = formState;

```

## 重置表单
重置表单，包括数据和状态信息
```tsx

const {reset} = form;
<Button type="button" className="mt-2" onClick={() => {  
  reset()  
}}>Reset</Button>

```

## 验证模式
- onSubmit：提交时校验
- onChange：输入时校验
- onTouched：用户交互时校验
- onBlur：失去焦点时校验
- all：所有情况都校验
```tsx
const form = useForm<FormValues>({  
  defaultValues: defaultFormValues,  
  mode: "onSubmit"  
});
```

## 结合 shadui 和 zod 使用
```tsx
"use client"  
import {useForm} from "react-hook-form";  
import {z} from "zod"  
import {zodResolver} from "@hookform/resolvers/zod";  
import {Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage} from "@/components/ui/form";  
import {Button} from "@/components/ui/button";  
import {Input} from "@/components/ui/input";  
  
type FormValues = {  
  email: string  
}  
  
const defaultValues: FormValues = {  
  email: ""  
}  
  
const formSchema = z.object({  
  email: z.string().nonempty("email required").email("email invalid")  
})  
  
export default function ShadUIForm() {  
  
  const form = useForm<FormValues>({  
    defaultValues,  
    resolver: zodResolver(formSchema),  
  })  
  const {handleSubmit} = form;  
  const onSubmit = (data: FormValues) => {  
    console.log(data)  
  }  
  
  return (<div className="w-full h-full flex justify-center items-center">  
    <Form {...form}>  
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-8 w-[300px]">  
        <FormField  
          control={form.control}  
          name="email"  
          render={({field}) => (  
            <FormItem>  
              <FormLabel>Email</FormLabel>  
              <FormControl>  
                <Input placeholder="enter email" {...field} />  
              </FormControl>  
              <FormDescription>  
                This is your public display name.  
              </FormDescription>  
              <FormMessage/>  
            </FormItem>  
          )}  
        />  
        <Button type="submit">Submit</Button>  
      </form>  
    </Form>  
  </div>)  
}
```
