import { useState } from "react"
import { useNavigate } from 'react-router-dom';

function Search() {
    const [searchInput, setSearchInput] = useState('')
    const [error,setError]=useState(false)
    const navigateTo = useNavigate();

function onSearch() {
        if(searchInput.trim()!==''){
            setError(false)
            console.log(searchInput)
            navigateTo(`/searchResults`,{state: { key: searchInput }})
            
            return;
        }
        else{
            setError(true)
            return;
        }
}


return (

        <div className="flex items-center justify-center h-screen">
            
            <div className="flex flex-col items-center bg-black p-8 rounded shadow-md">
            <h1 className="text-5xl font-bold mb-4 text-white">A Computer Science Archive</h1>
                <input
                    type="text"
                    className="border  p-2 mb-4 w-full"
                    placeholder="Enter text..."
                    onChange={(e) => setSearchInput(e.target.value)}
                />
                <div className="flex mb-4">
                    <button className="bg-blue-500 hover:bg-blue-700 text-white ml-2 mr-2 font-bold py-2 px-4 rounded-l" onClick={onSearch}>
                        Semantic Search
                    </button>
                    <button className="bg-green-500 hover:bg-green-700 text-white ml-2 mr-2 font-bold py-2 px-4 rounded-r" onClick={onSearch}>
                        Syntactic Search
                    </button>
                </div>
                {error && <p className="text-red-500 mb-4">*Please Enter Some Text</p>}
            </div>
        </div>
    )
}

export default Search