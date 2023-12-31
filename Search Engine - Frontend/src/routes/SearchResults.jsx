import { useEffect, useState ,useRef} from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import Card from "../Card";
import { ClusterCard } from "../ClusterCard";
import { Routes, Route, useParams } from 'react-router-dom';
import SearchSuggestions from "../components/suggestions";

export default function searchResults() {
    const searchInput = useLocation();
    const [searchQuery,setSearchQuery]=useState(searchInput?.state?.key)
    const [clusters, setClusters] = useState([])
    const [docs, setDocs] = useState([])
    const [error, setError] = useState(null)
    const [searchInputEle,setSearchInputEle]=useState('')
      // append this search input(searchInput.state.key) to the api
    const navigateTo = useNavigate();
    const SemanticSearchRef = useRef(null)
    const SyntacticSearchRef = useRef(null)
    let SearchType = 1
    const fetchClustersData = async () => {
        let url = `http://localhost:8000/semantic_search`;
        try {
            const response = await fetch(url, {
                method: "post",
                headers: {
                  'Accept': 'application/json',
                  'Content-Type': 'application/json'
                },
                //make sure to serialize your JSON body
                body: JSON.stringify({
                  'query' : searchQuery,
                  'type' : SearchType
                })
              })
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const result = await response.json()
            console.log(result)
            setClusters(result.clusterss)
            setDocs(result.docss)
            // console.log(result.docss)
        }
        catch (e) {
            setError(e)
        }
    }

    
    async function onSearch(buttonRef) {
        if (searchQuery.trim() !== '') {
            setError(false)

            if(buttonRef.current.innerText=='Semantic Search'){
                SearchType = 1

            }
            else if(buttonRef.current.innerText=='Syntactic Search'){
                SearchType = 0

            }
            navigateTo('/searchResults/')
            await fetchClustersData()
            // fetchDocsData()
            
            navigateTo('/searchResults/0') 
            //window.location.reload();

            console.log(searchQuery)
            return;
        }
        else {
            setError(true)
            return;
        }
    }

    function onInputSearchChange(item){
        setSearchQuery(item.suggestion)
        setSearchInputEle('')

    }
    
    function onInputChange(e){
        
        setSearchQuery(e.target.value)
        setSearchInputEle(e.target.value)
        //navigateTo('/searchResults/')
        if(e.target.value!=''){
            setError(false)
        }else{
            setError(true)
        }
    }
     // set the api endpoint to this url 
    useEffect(()=>{
        (async()=>{
            await fetchClustersData();

            navigateTo('/searchResults/0')
        })()

         

     },[])



    return (
        <div className="">
            <div className="flex mt-4">
                <div id="sidebar" className=" flex-shrink-0  bg-gray-900 text-white p-4 ">
                    <h1 className="text-2xl font-bold mb-4 text-white">Search</h1>
                   
                    <input
                        id="q"
                        aria-label="Search contacts"
                        placeholder="Search"
                        type="search"
                        name="q"
                        value={searchQuery}
                        className="bg-gray-700 text-white p-2 rounded w-full"
                        onChange={(e)=>onInputChange(e)}
                    />
                    
                    <SearchSuggestions clusterSearchInput={searchInputEle} onInputSearchChange={onInputSearchChange}/>
                    {error && <p className="text-red-500 mb-4">*Please Enter Some Text</p>}
                    <div id="search-spinner" aria-hidden hidden={true} className="spinner"></div>
                    <div className="sr-only" aria-live="polite"></div>
                    <div className=" mt-2 mb-4 flex">
                        <button type="submit" className=" ml-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" ref={SemanticSearchRef} onClick={()=>{onSearch(SemanticSearchRef)}}>
                            Semantic Search
                        </button>
                        <button type="submit" className=" ml-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" ref={SyntacticSearchRef} onClick={()=>{onSearch(SyntacticSearchRef)}}>
                            Syntactic Search
                        </button>
                    </div>
                    <nav>
                        <ul className=" mt-2 mb-4 ">
                            {clusters?.map((cluster) => <Link  to={`${cluster.clusterId}`}><ClusterCard key={cluster.clusterName} clusterId={cluster.clusterId} clusterName={cluster.clusterName} clusterLength={cluster.documentList.length} /></Link>)}
                        </ul>
                    </nav>
                </div>
                <div id='detail' className='flex-grow'>
                    <Outlet context={[docs]} />
                </div>
            </div>
        </div>
    )
}