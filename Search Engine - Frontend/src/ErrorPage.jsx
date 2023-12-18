import { useRouteError } from "react-router-dom";

export default function ErrorPage() {
    const error = useRouteError();
    console.error(error);

    return (
        <div className="flex items-center justify-center h-screen bg-black  text-white" id="error-page" >
            <div className="text-center">
                <h1 className="text-4xl font-bold text-red-500 mb-4" >Oops!</h1>
                <p className="text-2xl font-bold text-white-500 mb-4">Sorry, an unexpected error has occurred.</p>
                <p className="text-2xl font-bold text-white-500 mb-4">
                    <i>{error.statusText || error.message}</i>
                </p>
            </div>

        </div>
    );
}