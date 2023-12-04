import { useState } from 'react'


import './App.css'

function App() {
  const [searchInput,setSearchInput]=useState('')

  function onSymanticSearch(){
    
  }

  function onSyntacticSearch(){

  }

  return (
    
<div className="flex items-center justify-center h-screen">
  <div className="flex flex-col items-center bg-black p-8 rounded shadow-md">
    <input
      type="text"
      className="border  p-2 mb-4 w-full"
      placeholder="Enter text..."
      onChange={()=>setSearchInput(e.target.value)}
    />
    <div className="flex mb-4">
      <button className="bg-blue-500 hover:bg-blue-700 text-white ml-2 mr-2 font-bold py-2 px-4 rounded-l" onClick={onSymanticSearch}>
        SymanticSearch
      </button>
      <button className="bg-green-500 hover:bg-green-700 text-white ml-2 mr-2 font-bold py-2 px-4 rounded-r" onClick={onSyntacticSearch}>
        SyntacticSearch
      </button>
    </div>
    <p className="text-red-500 mb-4">Error message goes here</p>
  </div>
</div>    
  )
}

export default App






