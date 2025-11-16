import { useState } from 'react'
import './App.css'

function App() {

  const [isTyping, setIsTyping] = useState(false);
  const [content, setContent] = useState("");

  const handleChange = (event) => {
    setContent(event.target.value); // Cập nhật state 'content' với giá trị mới từ textarea
    if (event.target.value.length > 0) { // Kiểm tra độ dài của giá trị mới
      setIsTyping(true);
    } else {
      setIsTyping(false);
    }
  }

  return (
    <div className="px-[200px]">

      <div className="flex justify-between items-center z-100 px-3 py-5 border-b border-gray-600 cursor-pointer">
        <div className="text-lg align-center">
          <p>Chatbot health care</p>
        </div>

        <div className='text-4xl'>
          <i className="fa-regular fa-circle-user"></i>
        </div>
      </div>

      <div className=' flex flex-col gap-10 items-center'>

        <div className='flex justify-center mt-[200px]'>
          <span className="ombre-color">Can i help you, sir!</span>
        </div>

        <div className="w-[760px] border border-gray-600 rounded-2xl px-[20px] py-[15px] shadow relative bottom-[-180px]">

          <textarea value={content} onChange={handleChange} id="input" rows="2" placeholder="Hỏi Gemini"></textarea>

          <div className="textbox--bottom">
            <div className="textbox--bottom-left">

              <div id="plus-icon">
                <i className=" fa-solid fa-plus"></i>

                <div id="popup-menu">
                  <ul>
                    <li>
                      <i className="fa-regular fa-image"></i>
                      Hình ảnh
                    </li>
                    <li >
                      <i className="fa-solid fa-paperclip"></i>
                      Tệp
                    </li>
                  </ul>
                </div>
              </div>

              <div className="deepSearch">
                <i className="fa-solid fa-magnifying-glass mx-2"></i>
                <span>Deep search</span>
              </div>
            </div>

            <div className="textbox--bottom-right">
              {
                isTyping
                  ? <i className="fa-solid fa-paper-plane"></i>
                  : <i className="micro-icon icon fa-solid fa-microphone"></i>
              }
            </div>

          </div>
        </div>
      </div>


    </div>
  )
}

export default App
